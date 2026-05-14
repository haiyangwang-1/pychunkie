"""Operator assembly, matrix-free application, and evaluation."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator

from . import lege
from ._layout import (
    as_boundary_chunk_tensor,
    as_boundary_field_matrix,
    as_boundary_point_matrix,
    as_boundary_vector,
    as_boundary_weight_matrix,
    boundary_component_weights,
    density_matmul_argument,
    weighted_density_for_boundary,
)
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
    raw_options: dict[str, Any] | None,
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    l2scale: bool | None = None,
    dval: ArrayLike | float | complex | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    flam_occupancy: int | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    force_adaptive: bool | None = None,
    corrections: bool | None = None,
    correction_matrix: ArrayLike | sparse.spmatrix | None = None,
    side: str | None = None,
    near_factor: float | None = None,
    rcip: Any | None = None,
    return_rcip: bool | None = None,
    rcip_context: Any | None = None,
    rcip_subdivisions: int | float | None = None,
    rcip_save_depth: int | None = None,
    rcip_eval_depth: int | None = None,
    rcip_vertices: ArrayLike | None = None,
    rcip_ignore_vertices: ArrayLike | None = None,
) -> dict[str, Any]:
    if raw_options is None:
        options: dict[str, Any] = {}
    elif bool(raw_options.get(_NORMALIZED_OPTIONS_MARKER, False)):
        options = dict(raw_options)
    else:
        warnings.warn(
            "operator option dictionaries are deprecated; use keyword-only arguments instead",
            DeprecationWarning,
            stacklevel=3,
        )
        options = dict(raw_options)
    _set_option(options, "acceleration", acceleration)
    _set_option(options, "usepquad", use_panel_quadrature)
    _set_option(options, "l2scale", l2scale)
    _set_option(options, "dval", dval)
    _set_option(options, "tol", tol)
    _set_option(options, "flamtype", flam_type)
    _set_option(options, "occ", flam_occupancy)
    _set_option(options, "rank_or_tol", rank_or_tol)
    _set_option(options, "useproxy", proxy)
    _set_option(options, "forceadap", force_adaptive)
    _set_option(options, "corrections", corrections)
    _set_option(options, "cormat", correction_matrix)
    _set_option(options, "side", side)
    _set_option(options, "fac", near_factor)
    _set_option(options, "rcip", rcip)
    _set_option(options, "return_rcip", return_rcip)
    _set_option(options, "rcip_context", rcip_context)
    _set_option(options, "nsub", rcip_subdivisions)
    _set_option(options, "rcip_savedepth", rcip_save_depth)
    _set_option(options, "rcip_eval_depth", rcip_eval_depth)
    _set_option(options, "rcip_vertices", rcip_vertices)
    _set_option(options, "rcip_ignore_vertices", rcip_ignore_vertices)
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
    def from_any(cls, options: dict[str, Any] | _OperatorOptions | None) -> _OperatorOptions:
        if isinstance(options, cls):
            return options
        return cls({} if options is None else dict(options))

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
        raw_options = {
            "verb": int(self.flag("verb")),
            "lvlmax": self.raw.get("lvlmax", np.inf),
        }
        if store_default is not None:
            raw_options["store"] = self.raw.get("store", store_default)
        return raw_options

    def fmm_tol(self, default: float = 1.0e-12) -> float:
        return float(self.raw.get("eps", self.raw.get("tol", default)))

    def uses_special_quadrature(self, kernel: Callable[[Any, Any], np.ndarray]) -> bool:
        if self.flag("forcesmooth") or self.flag("usesmooth"):
            return False
        if self.flag("forceadap"):
            return True
        return getattr(kernel, "sing", "") in {"log", "pv", "hs"}

    def special_quadrature_type(self, kernel: Callable[[Any, Any], np.ndarray]) -> str:
        qtype = str(self.raw.get("sing", getattr(kernel, "sing", "log") or "log")).lower()
        return "log" if qtype == "smooth" else qtype


class ChunkerFMMMatrix(LinearOperator):
    """Matrix-free ``chunkermat`` operator using kernel FMM application."""

    def __init__(
        self,
        chunker: Chunker,
        kernel: Callable[[Any, Any], np.ndarray],
        options: dict[str, Any] | None = None,
    ):
        self.kernel = kernel
        self.options = {} if options is None else dict(options)
        self._correction_mat: spmatrix | None = None
        if _is_block_kernel_matrix(kernel):
            self.chunker = chunker
            self.block_layout = _block_kernel_layout(chunker, kernel)
            self.opdims = None
            for item in self.block_layout.kernels.flat:
                _require_fmm(item)
            dtype = _block_operator_dtype(self.block_layout)
            shape = (int(self.block_layout.row_offsets[-1]), int(self.block_layout.col_offsets[-1]))
        else:
            self.chunker = _require_chunker(chunker)
            self.block_layout = None
            self.opdims = _kernel_opdims(self.chunker, kernel)
            _require_fmm(kernel)
            dtype = _operator_dtype(self.chunker, kernel)
            shape = (
                self.chunker.npt * int(self.opdims[0]),
                self.chunker.npt * int(self.opdims[1]),
            )
        super().__init__(dtype=dtype, shape=shape)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.shape[1]:
            raise ValueError("density has incompatible size")
        if self.block_layout is not None:
            return _block_chunkermatapply_fmm(
                self.block_layout, x, self.options, self._correction()
            )
        return _chunkermatapply_fmm(self.chunker, self.kernel, x, self.options, self._correction())

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
            if not _block_uses_special_quadrature(self.block_layout, self.options):
                return None
            if self._correction_mat is None:
                self._correction_mat = _block_special_correction_matrix(
                    self.block_layout, self.options
                )
            return self._correction_mat
        if not _uses_special_quadrature(self.kernel, self.options):
            return None
        if self._correction_mat is None:
            self._correction_mat = _special_correction_matrix(
                self.chunker, self.kernel, self.options
            )
        return self._correction_mat


class ChunkerFLAMMatrix(LinearOperator):
    """Matrix-free ``chunkermat`` operator backed by a PyFLAM factorization."""

    def __init__(
        self,
        chunker: Chunker,
        kernel: Callable[[Any, Any], np.ndarray],
        dval: ArrayLike | float | complex = 0.0,
        options: dict[str, Any] | None = None,
    ):
        self.chunker = _require_chunker(chunker) if not _is_block_kernel_matrix(kernel) else chunker
        self.kernel = kernel
        self.options = {} if options is None else dict(options)
        self.factor = chunkerflam(self.chunker, self.kernel, dval, self.options)
        self.flamtype = _flamtype(self.options)
        if _is_block_kernel_matrix(kernel):
            self.block_layout = _block_kernel_layout(chunker, kernel)
            self.opdims = None
            dtype = _block_operator_dtype(self.block_layout)
            shape = (int(self.block_layout.row_offsets[-1]), int(self.block_layout.col_offsets[-1]))
        else:
            self.block_layout = None
            self.opdims = _kernel_opdims(self.chunker, kernel)
            dtype = _operator_dtype(self.chunker, kernel)
            shape = (
                self.chunker.npt * int(self.opdims[0]),
                self.chunker.npt * int(self.opdims[1]),
            )
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
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    dval: ArrayLike | float | complex = 0.0,
    options: dict[str, Any] | None = None,
    *,
    use_panel_quadrature: bool | None = None,
    l2scale: bool | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    flam_occupancy: int | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    side: str | None = None,
):
    """Build a PyFLAM compressed representation of a chunker system matrix.

    ``chunker`` may be a single chunker, a chunkgraph-like object, or an
    explicit sequence of edge chunkers when ``kernel`` is an edge-by-edge block
    kernel matrix. ``dval`` adds a diagonal shift to the discretized operator.
    Important keyword options include ``flam_type`` (``"rskelf"`` or
    ``"rskel"``), ``flam_occupancy``, ``rank_or_tol``/``tol``, ``proxy``,
    and ``l2scale``.
    """

    pyflam = _require_pyflam()
    operator_options = _normalize_public_options(
        options,
        use_panel_quadrature=use_panel_quadrature,
        l2scale=l2scale,
        tol=tol,
        flam_type=flam_type,
        flam_occupancy=flam_occupancy,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        side=side,
    )
    if _is_block_kernel_matrix(kernel):
        return _chunkerflam_block(chunker, kernel, dval, operator_options, pyflam)

    boundary = _require_chunker(chunker)
    opdims = _kernel_opdims(boundary, kernel)
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    nrows = boundary.npt * op0
    ncols = boundary.npt * op1
    if nrows != ncols:
        raise ValueError("chunkerflam requires a square discretized operator")

    dval_vec = _dval_vector(dval, nrows)
    spmat = operator_options.get("sp_nonsmooth", None)
    if spmat is None:
        spmat = _special_overwrite_matrix(boundary, kernel, operator_options)
    else:
        spmat = spmat.tocsr() if sparse.issparse(spmat) else sparse.csr_matrix(spmat)
    has_dval = bool(np.any(dval_vec != 0))

    from .acceleration import flam

    l2scale = _l2scale(operator_options)

    def matfun(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        out = flam.kernbyindex(rows, cols, boundary, kernel, (op0, op1), spmat, l2scale)
        if has_dval:
            out = _add_diagonal_shift(out, rows, cols, dval_vec)
        return out

    srcinfo = PointInfo.from_any(boundary)
    xflam = np.repeat(np.real(srcinfo.r), op1, axis=1)
    flamtype = _flamtype(operator_options)
    occ = _flam_occ(operator_options)
    rank_or_tol = _flam_rank_or_tol(operator_options)
    opts_flam = _flam_options(operator_options)
    useproxy = _flag(operator_options, "useproxy", True) and boundary.datadim == 0 and op0 == op1
    pxyfun = None
    if useproxy and flamtype == "rskelf":
        pxyfun = _chunkerflam_proxyfun(boundary, kernel, (op0, op1), operator_options)
    elif useproxy and flamtype == "rskel":
        pxyfun = _chunkerkerneval_proxyfun(
            boundary, kernel, PointInfo.from_any(boundary), (op0, op1), operator_options
        )

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
        out = flam.kernbyindex(
            rows, cols, layout.chunkers, layout.kernels, layout.opdims_mat, spmat, l2scale
        )
        if has_dval:
            out = _add_diagonal_shift(out, rows, cols, dval_vec)
        return out

    xflam = np.column_stack(
        [
            np.repeat(np.real(PointInfo.from_any(chunker).r), int(opdim), axis=1)
            for chunker, opdim in zip(layout.chunkers, layout.coldims, strict=True)
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
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any] | None = None,
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    l2scale: bool | None = None,
    dval: ArrayLike | float | complex | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    flam_occupancy: int | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    side: str | None = None,
    rcip: Any | None = None,
    return_rcip: bool | None = None,
    rcip_context: Any | None = None,
    rcip_subdivisions: int | float | None = None,
    rcip_save_depth: int | None = None,
    rcip_vertices: ArrayLike | None = None,
    rcip_ignore_vertices: ArrayLike | None = None,
) -> (
    np.ndarray
    | ChunkerFMMMatrix
    | ChunkerFLAMMatrix
    | ChunkerRCIPMatrix
    | tuple[np.ndarray, RCIPContext]
):
    """Assemble or factor a boundary-integral operator on a chunker-like object.

    The default path returns a dense NumPy matrix using native or special
    quadrature. ``acceleration="fmm"`` returns a matrix-free
    :class:`ChunkerFMMMatrix` for kernels with an FMM evaluator.
    ``acceleration="flam"`` returns a :class:`ChunkerFLAMMatrix`
    backed by PyFLAM and accepts ``dval`` for second-kind or shifted systems.
    Block kernel matrices are supported for explicit chunker sequences and
    chunkgraphs.
    """

    operator_options = _normalize_public_options(
        options,
        acceleration=acceleration,
        quadrature=quadrature,
        use_panel_quadrature=use_panel_quadrature,
        l2scale=l2scale,
        dval=dval,
        tol=tol,
        flam_type=flam_type,
        flam_occupancy=flam_occupancy,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        side=side,
        rcip=rcip,
        return_rcip=return_rcip,
        rcip_context=rcip_context,
        rcip_subdivisions=rcip_subdivisions,
        rcip_save_depth=rcip_save_depth,
        rcip_vertices=rcip_vertices,
        rcip_ignore_vertices=rcip_ignore_vertices,
    )
    if _is_block_kernel_matrix(kernel):
        acceleration = _acceleration(operator_options)
        if acceleration == "flam":
            return ChunkerFLAMMatrix(
                chunker, kernel, operator_options.get("dval", 0.0), operator_options
            )
        if acceleration == "fmm":
            return ChunkerFMMMatrix(chunker, kernel, operator_options)
        return _block_kernel_mat(chunker, kernel, operator_options)

    if _chunkgraph_rcip_mat_enabled(chunker, kernel, operator_options):
        mat, context = _chunkgraph_rcip_mat(chunker, kernel, operator_options)
        if _flag(operator_options, "return_rcip"):
            return mat, context
        return mat
    if (
        _is_chunkgraph_like(chunker)
        and "rcip" in operator_options
        and not _rcip_option_enabled(operator_options)
    ):
        try:
            chunker._last_rcip_context = None
        except AttributeError:
            pass

    boundary = _require_chunker(chunker)
    acceleration = _acceleration(operator_options)
    if acceleration == "flam":
        return ChunkerFLAMMatrix(
            boundary, kernel, operator_options.get("dval", 0.0), operator_options
        )
    if acceleration == "fmm":
        _require_fmm(kernel)
        return ChunkerFMMMatrix(boundary, kernel, operator_options)
    if _uses_special_quadrature(kernel, operator_options):
        if _flag(operator_options, "adaptive_correction"):
            from .quadrature import adaptive as quadadap

            adap_options = dict(operator_options)
            adap_options.setdefault("sing", _special_quadrature_type(kernel, operator_options))
            adap_options.setdefault("usepquad", _boundary_pquad_enabled(operator_options))
            mat = quadadap.buildmat(boundary, kernel, getattr(kernel, "opdims", None), adap_options)
        else:
            from .quadrature import ggq as quadggq

            mat = quadggq.buildmat(
                boundary,
                kernel,
                getattr(kernel, "opdims", None),
                _special_quadrature_type(kernel, operator_options),
                pquad_side=_pquad_side(operator_options),
                usepquad=_boundary_pquad_enabled(operator_options),
            )
        return _apply_l2scale_matrix(boundary, mat) if _l2scale(operator_options) else mat

    srcinfo = PointInfo.from_any(boundary)
    mat = _eval_kernel(kernel, srcinfo, srcinfo)
    wts = as_boundary_vector(boundary.wts, name="weights")
    if mat.shape[1] == boundary.npt:
        out = mat * wts[None, :]
        out = _apply_laplace_double_self_limit(boundary, kernel, out)
        out = _apply_laplace_sprime_self_limit(boundary, kernel, out)
        return _apply_l2scale_matrix(boundary, out) if _l2scale(operator_options) else out
    if mat.shape[1] % boundary.npt != 0:
        raise ValueError("kernel column dimension is incompatible with chunker points")
    opdims_col = mat.shape[1] // boundary.npt
    out = mat * np.repeat(wts, opdims_col)[None, :]
    out = _apply_stokes_strac_self_limit(boundary, kernel, out)
    return _apply_l2scale_matrix(boundary, out) if _l2scale(operator_options) else out


def chunkermatapply(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    density: ArrayLike,
    options: dict[str, Any] | None = None,
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    l2scale: bool | None = None,
    dval: ArrayLike | float | complex | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    flam_occupancy: int | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    side: str | None = None,
    rcip: Any | None = None,
    rcip_context: Any | None = None,
    rcip_subdivisions: int | float | None = None,
    rcip_save_depth: int | None = None,
    rcip_vertices: ArrayLike | None = None,
    rcip_ignore_vertices: ArrayLike | None = None,
) -> np.ndarray:
    """Apply ``chunkermat(chunker, kernel, ...)`` without forcing dense materialization."""

    operator_options = _normalize_public_options(
        options,
        acceleration=acceleration,
        quadrature=quadrature,
        use_panel_quadrature=use_panel_quadrature,
        l2scale=l2scale,
        dval=dval,
        tol=tol,
        flam_type=flam_type,
        flam_occupancy=flam_occupancy,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        side=side,
        rcip=rcip,
        rcip_context=rcip_context,
        rcip_subdivisions=rcip_subdivisions,
        rcip_save_depth=rcip_save_depth,
        rcip_vertices=rcip_vertices,
        rcip_ignore_vertices=rcip_ignore_vertices,
    )
    acceleration = _acceleration(operator_options)
    if acceleration == "fmm":
        if _is_block_kernel_matrix(kernel):
            for item in np.asarray(kernel, dtype=object).flat:
                _require_fmm(item)
        else:
            _require_fmm(kernel)
    chunker_like = (
        chunker
        if _is_block_kernel_matrix(kernel)
        or _chunkgraph_rcip_mat_enabled(chunker, kernel, operator_options)
        else _require_chunker(chunker)
    )
    op = chunkermat(chunker_like, kernel, operator_options)
    density_arg = _density_matmul_arg(op.shape[1], density)
    return op @ density_arg


def chunkerintegral(
    chunker: Chunker,
    integrand: Callable[[np.ndarray], ArrayLike] | ArrayLike,
    options: dict[str, Any] | None = None,
) -> float:
    """Integrate scalar values over a chunker with the native smooth rule."""

    _ = options
    boundary = _require_chunker(chunker)
    if callable(integrand):
        vals = np.asarray(
            integrand(
                as_boundary_point_matrix(boundary.r, boundary.dim, boundary.npt, name="positions")
            )
        )
    else:
        vals = np.asarray(integrand)
    if vals.size != boundary.npt:
        raise ValueError("integrand must evaluate to one scalar value per chunker point")
    return float(
        np.dot(
            as_boundary_vector(boundary.wts, name="weights"),
            as_boundary_vector(vals, name="integrand values"),
        )
    )


def chunkerinterior(
    chunker: Chunker,
    points: Chunker | dict[str, Any] | ArrayLike | tuple[ArrayLike, ArrayLike] | list[ArrayLike],
    options: dict[str, Any] | None = None,
    *,
    acceleration: str | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    flam_occupancy: int | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
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

    boundary = _require_chunker(chunker)
    operator_options = _normalize_public_options(
        options,
        acceleration=acceleration,
        tol=tol,
        flam_type=flam_type,
        flam_occupancy=flam_occupancy,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        near_factor=near_factor,
    )
    _set_option(operator_options, "closecorr", close_correction)
    _set_option(operator_options, "axissym", axis_symmetric)
    if boundary.dim != 2:
        raise ValueError("interior only well-defined for 2D chunkers")

    grid_shape = None
    if isinstance(points, (tuple, list)) and len(points) == 2:
        x = np.asarray(points[0], dtype=float)
        y = np.asarray(points[1], dtype=float)
        xx, yy = np.meshgrid(x, y)
        pts = np.vstack((xx.ravel(), yy.ravel()))
        grid_shape = xx.shape
    elif _as_chunker(points) is not None:
        target_boundary = _require_chunker(points)
        pts = as_boundary_point_matrix(
            target_boundary.r,
            target_boundary.dim,
            target_boundary.npt,
            name="target positions",
        )
    elif isinstance(points, dict) and "r" in points:
        arr = np.asarray(points["r"], dtype=float)
        pts = arr.reshape(arr.shape[0], -1)
    else:
        arr = np.asarray(points, dtype=float)
        pts = arr.reshape(arr.shape[0], -1)

    if pts.shape[0] != 2:
        raise ValueError("target points must be two-dimensional")

    acceleration = _acceleration(operator_options)
    if acceleration in {"fmm", "flam"}:
        from .kernels import kernel

        lap_d = kernel("lap", "d")
        density = np.ones(boundary.npt)
        eval_options = dict(operator_options)
        eval_options["acceleration"] = acceleration
        vals = chunkerkerneval(
            boundary,
            lap_d,
            density,
            PointInfo(r=pts),
            eval_options,
        )
        vals = as_boundary_vector(vals, name="interior layer values")
        inside = vals < -0.5
        if _flag(operator_options, "closecorr", _flag(operator_options, "corrections", True)):
            near_fac = float(operator_options.get("near_fac", operator_options.get("fac", 1.0)))
            near = np.any(boundary.flagnear(pts, fac=near_fac), axis=1)
            if np.any(near):
                corrected = _chunkerinterior_direct(
                    boundary, pts[:, near], _flag(operator_options, "axissym")
                )
                inside[near] = corrected
    else:
        inside = _chunkerinterior_direct(boundary, pts, _flag(operator_options, "axissym"))

    if grid_shape is not None:
        return inside.reshape(grid_shape)
    return inside


def _chunkerinterior_direct(
    chunker: Chunker, points: np.ndarray, axissym: bool = False
) -> np.ndarray:
    inside = np.zeros(points.shape[1], dtype=bool)
    for boundary in _chunker_component_polygons(chunker, axissym):
        inside ^= _points_in_polygon(points, boundary)
    return inside


def chunkerkerneval(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    density: ArrayLike,
    target: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any] | None = None,
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    flam_occupancy: int | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    force_adaptive: bool | None = None,
    correction_matrix: ArrayLike | sparse.spmatrix | None = None,
    near_factor: float | None = None,
    side: str | None = None,
    rcip: Any | None = None,
    rcip_context: Any | None = None,
    rcip_subdivisions: int | float | None = None,
    rcip_save_depth: int | None = None,
    rcip_eval_depth: int | None = None,
) -> np.ndarray:
    """Evaluate a layer potential generated by a chunker density at targets.

    ``density`` is converted to the flat boundary vector expected by solver and
    backend adapters, including any vector components implied by
    ``kernel.opdims``. ``target`` may be raw coordinates, a
    ``PointInfo``/mapping, another chunker, or a chunkgraph-like object.
    Options include ``acceleration="fmm"`` for supported FMM kernels,
    ``acceleration="flam"`` for PyFLAM target evaluation, and ``force_adaptive=True``
    for adaptive close-target correction.
    """

    operator_options = _normalize_public_options(
        options,
        acceleration=acceleration,
        quadrature=quadrature,
        use_panel_quadrature=use_panel_quadrature,
        tol=tol,
        flam_type=flam_type,
        flam_occupancy=flam_occupancy,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        force_adaptive=force_adaptive,
        correction_matrix=correction_matrix,
        near_factor=near_factor,
        side=side,
        rcip=rcip,
        rcip_context=rcip_context,
        rcip_subdivisions=rcip_subdivisions,
        rcip_save_depth=rcip_save_depth,
        rcip_eval_depth=rcip_eval_depth,
    )
    same_source_target = target is chunker
    rcip_context = _chunkgraph_rcip_eval_context(chunker, operator_options)
    if rcip_context is not None and not same_source_target:
        return _chunkgraph_rcip_eval(
            chunker, kernel, density, target, operator_options, rcip_context
        )

    boundary = _require_chunker(chunker)
    acceleration = _acceleration(operator_options)
    if (
        same_source_target
        and _uses_special_quadrature(kernel, operator_options)
        and acceleration != "fmm"
    ):
        mat_options = dict(operator_options)
        mat_options.pop("acceleration", None)
        vals = chunkermat(boundary, kernel, mat_options) @ as_boundary_vector(
            density, name="density"
        )
        opdims = getattr(kernel, "opdims", (1, 1))[0]
        return as_boundary_field_matrix(vals, opdims, boundary.npt, name="kernel values")
    if acceleration == "flam":
        targinfo = PointInfo.from_any(target)
        if _flag(operator_options, "forceadap"):
            smooth_options = dict(operator_options)
            smooth_options.pop("forceadap", None)
            vals = as_boundary_vector(
                _chunkerkerneval_flam(boundary, kernel, density, targinfo, smooth_options),
                name="FLAM values",
            )
            correction_options = dict(operator_options)
            correction_options["recompute_source_normals"] = True
            correction_options.setdefault("transinv", False)
            correction = _target_adaptive_correction_matrix(
                boundary, kernel, targinfo, correction_options
            )
            vals = vals + correction @ as_boundary_vector(density, name="density")
            return as_boundary_field_matrix(
                vals,
                vals.size // targinfo.r.shape[1],
                targinfo.r.shape[1],
                name="corrected FLAM values",
            )
        return _chunkerkerneval_flam(boundary, kernel, density, targinfo, operator_options)
    use_fmm = acceleration == "fmm"
    if use_fmm:
        _require_fmm(kernel)

    srcinfo = PointInfo.from_any(boundary)
    targinfo = PointInfo.from_any(target)
    if same_source_target and _uses_special_quadrature(kernel, operator_options):
        smooth_options = _smooth_fmm_options(operator_options)
        smooth_options["acceleration"] = "fmm"
        vals = as_boundary_vector(
            chunkerkerneval(boundary, kernel, density, boundary, smooth_options),
            name="smooth FMM values",
        )
        vals = vals + _special_correction_matrix(
            boundary, kernel, operator_options
        ) @ as_boundary_vector(
            density,
            name="density",
        )
        return as_boundary_field_matrix(
            vals,
            vals.size // targinfo.r.shape[1],
            targinfo.r.shape[1],
            name="corrected FMM values",
        )
    cormat = operator_options.get("cormat", None)
    if cormat is not None:
        mat = _eval_kernel(kernel, srcinfo, targinfo)
        weighted = _weighted_density(boundary, density)
        density_vec = as_boundary_vector(density, name="density")
        corr_vals = (
            cormat @ density_vec if sparse.issparse(cormat) else np.asarray(cormat) @ density_vec
        )
        vals = mat @ weighted + corr_vals
        return as_boundary_field_matrix(
            vals,
            vals.size // targinfo.r.shape[1],
            targinfo.r.shape[1],
            name="corrected values",
        )
    if _flag(operator_options, "forceadap") and use_fmm:
        weighted = _weighted_density(boundary, density)
        vals = kernel.fmm(_fmm_tol(operator_options), srcinfo, targinfo, weighted)
        if isinstance(vals, tuple):
            vals = vals[0]
        vals = as_boundary_vector(vals, name="FMM values")
        correction_options = dict(operator_options)
        correction_options["recompute_source_normals"] = True
        correction_options.setdefault("transinv", False)
        correction = _target_adaptive_correction_matrix(
            boundary, kernel, targinfo, correction_options
        )
        vals = vals + correction @ as_boundary_vector(density, name="density")
        return as_boundary_field_matrix(
            vals,
            vals.size // targinfo.r.shape[1],
            targinfo.r.shape[1],
            name="corrected FMM values",
        )
    if _flag(operator_options, "forceadap"):
        eval_options = dict(operator_options)
        eval_options["recompute_source_normals"] = True
        eval_options.setdefault("transinv", False)
        mat = _target_adaptive_matrix(boundary, kernel, targinfo, eval_options)
        vals = mat @ as_boundary_vector(density, name="density")
        return as_boundary_field_matrix(
            vals,
            vals.size // targinfo.r.shape[1],
            targinfo.r.shape[1],
            name="adaptive values",
        )
    if use_fmm:
        weighted = _weighted_density(boundary, density)
        vals = kernel.fmm(_fmm_tol(operator_options), srcinfo, targinfo, weighted)
        if isinstance(vals, tuple):
            vals = vals[0]
        vals_arr = np.asarray(vals)
        return as_boundary_field_matrix(
            vals_arr,
            vals_arr.size // targinfo.r.shape[1],
            targinfo.r.shape[1],
            name="FMM values",
        )

    mat = _eval_kernel(kernel, srcinfo, targinfo)
    weighted = _weighted_density(boundary, density)
    vals = mat @ weighted
    return as_boundary_field_matrix(
        vals,
        vals.size // targinfo.r.shape[1],
        targinfo.r.shape[1],
        name="kernel values",
    )


def chunkerkernevalmat(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    target: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any] | None = None,
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    flam_occupancy: int | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    force_adaptive: bool | None = None,
    corrections: bool | None = None,
    near_factor: float | None = None,
    side: str | None = None,
) -> np.ndarray | spmatrix:
    """Build the matrix mapping source densities to target values.

    This is the materialized companion to :func:`chunkerkerneval`. It is useful
    for diagnostics, custom solvers, and adaptive correction matrices. Ordinary
    evaluation matrices are dense; ``corrections=True`` returns the sparse
    near-target correction matrix accepted by legacy ``cormat`` evaluation
    calls. With ``acceleration="fmm"``, this routine materializes the target
    evaluation matrix by applying the kernel FMM evaluator to basis densities.
    """

    same_source_target = target is chunker
    boundary = _require_chunker(chunker)
    operator_options = _normalize_public_options(
        options,
        acceleration=acceleration,
        quadrature=quadrature,
        use_panel_quadrature=use_panel_quadrature,
        tol=tol,
        flam_type=flam_type,
        flam_occupancy=flam_occupancy,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        force_adaptive=force_adaptive,
        corrections=corrections,
        near_factor=near_factor,
        side=side,
    )
    if _flag(operator_options, "corrections"):
        return _target_adaptive_correction_matrix(
            boundary, kernel, PointInfo.from_any(target), operator_options
        )
    acceleration = _acceleration(operator_options)
    if acceleration == "flam":
        if same_source_target and _uses_special_quadrature(kernel, operator_options):
            return np.asarray(chunkermat(boundary, kernel, operator_options))
        if _flag(operator_options, "forceadap"):
            targinfo = PointInfo.from_any(target)
            smooth_options = dict(operator_options)
            smooth_options.pop("forceadap", None)
            return (
                _chunkerkernevalmat_flam(boundary, kernel, targinfo, smooth_options)
                + _target_adaptive_correction_matrix(
                    boundary, kernel, targinfo, operator_options
                ).toarray()
            )
        return _chunkerkernevalmat_flam(boundary, kernel, target, operator_options)
    if acceleration == "fmm":
        _require_fmm(kernel)
        if same_source_target and _uses_special_quadrature(kernel, operator_options):
            return np.asarray(chunkermat(boundary, kernel, operator_options))
        return _chunkerkernevalmat_fmm(boundary, kernel, target, operator_options)
    if same_source_target and _uses_special_quadrature(kernel, operator_options):
        return chunkermat(boundary, kernel, operator_options)

    srcinfo = PointInfo.from_any(boundary)
    targinfo = PointInfo.from_any(target)
    if _flag(operator_options, "forceadap"):
        return _target_adaptive_matrix(boundary, kernel, targinfo, operator_options)
    mat = _eval_kernel(kernel, srcinfo, targinfo)
    wts = as_boundary_vector(boundary.wts, name="weights")
    if mat.shape[1] == boundary.npt:
        return mat * wts[None, :]
    if mat.shape[1] % boundary.npt != 0:
        raise ValueError("kernel column dimension is incompatible with chunker points")
    opdims_col = mat.shape[1] // boundary.npt
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
        items = list(as_boundary_vector(obj, name="chunker sequence"))
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


def _chunkgraph_rcip_mat_enabled(
    obj: Any, kernel: Callable[[Any, Any], np.ndarray], options: dict[str, Any]
) -> bool:
    if not _rcip_option_enabled(options):
        return False
    if not _is_chunkgraph_like(obj):
        return False
    if _acceleration(options) != "dense" or _l2scale(options):
        return False
    if not _is_rcip_second_kind_kernel(kernel):
        return False
    merged = _as_chunker(obj)
    if merged is None:
        return False
    try:
        if _kernel_opdims(merged, kernel) != (1, 1):
            return False
    except _KERNEL_PROBE_EXCEPTIONS:
        return False
    return bool(_chunkgraph_nonsmooth_vertices(obj, options))


def _is_rcip_second_kind_kernel(kernel: Callable[[Any, Any], np.ndarray]) -> bool:
    name = str(getattr(kernel, "name", "")).lower()
    typ = str(getattr(kernel, "type", "")).lower()
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
    kernel: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any],
) -> tuple[ChunkerRCIPMatrix, RCIPContext]:
    from . import rcip

    merged = cg.merged()
    base_options = _strip_rcip_options(options)
    mat = np.asarray(chunkermat(merged, kernel, base_options)).copy()
    nsub = _rcip_nsub(options)
    savedepth = _rcip_savedepth(options, nsub)
    saved: list[Any] = []

    for ivert in _chunkgraph_nonsmooth_vertices(cg, options):
        edges, signs = cg.vstruc[ivert]
        edges = np.asarray(edges, dtype=int).reshape(-1)
        signs = np.asarray(signs, dtype=int).reshape(-1)
        isstart = signs < 0
        edge_chunks = _rcip_corner_edge_chunks(cg, edges, signs)
        pbc, pwbc, star_l, circ_l, star_s, circ_s, *_ = rcip.setup(cg.k, 1, edges.size, isstart)
        rmat, rcipsav = rcip.Rcompchunk(
            cg.echnks,
            edge_chunks,
            kernel,
            1,
            cg.verts[:, ivert],
            Pbc=pbc,
            PWbc=pwbc,
            starL=star_l,
            circL=circ_l,
            starS=star_s,
            circS=circ_s,
            options={"nsub": nsub, "rcip_savedepth": savedepth},
        )
        starind = _rcip_corner_star_indices(cg, edges, signs)
        rcipsav.starind = starind
        replacement = np.linalg.inv(rmat) - np.eye(rmat.shape[0], dtype=rmat.dtype)
        mat[np.ix_(starind, starind)] = replacement
        saved.append(rcipsav)

    context = RCIPContext(cg, kernel, saved, nsub, savedepth, 1)
    try:
        cg._last_rcip_context = context
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
    for edge, sign in zip(np.asarray(edges, dtype=int), np.asarray(signs, dtype=int), strict=True):
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
    for idx, (edge, sign) in enumerate(zip(edge_arr, sign_arr, strict=True)):
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
    kernel: Callable[[Any, Any], np.ndarray],
    density: ArrayLike,
    target: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any],
    context: RCIPContext,
) -> np.ndarray:
    from . import rcip

    if not context.saved:
        return chunkerkerneval(cg.merged(), kernel, density, target, _strip_rcip_options(options))

    dens_vec = as_boundary_vector(density, name="density")
    merged = cg.merged()
    if dens_vec.size != merged.npt * int(context.ndim):
        raise ValueError("density has incompatible size for RCIP context")

    rho_coarse = dens_vec.copy()
    for rcipsav in context.saved:
        starind = np.asarray(rcipsav.starind, dtype=int).reshape(-1)
        rho_coarse[starind] = 0.0

    eval_options = _strip_rcip_options(options)
    targinfo = PointInfo.from_any(target)
    vals = as_boundary_vector(
        chunkerkerneval(merged, kernel, rho_coarse, targinfo, eval_options),
        name="RCIP coarse values",
    )
    ndepth = int(options.get("rcip_eval_depth", options.get("rcip_ndepth", context.savedepth)))

    for rcipsav in context.saved:
        starind = np.asarray(rcipsav.starind, dtype=int).reshape(-1)
        rhocells, srcinfos, wtscells = rcip.rhohatInterp(dens_vec[starind], rcipsav, ndepth)
        local_targets = _shift_pointinfo(targinfo, np.asarray(rcipsav.ctr)[:, [0]])
        if _flag(eval_options, "forceadap"):
            for rho, srcinfo, wts in zip(rhocells, srcinfos, wtscells, strict=True):
                if srcinfo is None or wts is None:
                    continue
                local_chunker = _chunker_from_pointinfo(srcinfo, wts, rcipsav.k)
                vals = vals + as_boundary_vector(
                    chunkerkerneval(local_chunker, kernel, rho, local_targets, eval_options),
                    name="RCIP local values",
                )
        else:
            srcinfo = _merge_pointinfos([src for src in srcinfos if src is not None])
            if srcinfo is None:
                continue
            rho_local = np.concatenate(
                [as_boundary_vector(rho, name="RCIP density") for rho in rhocells]
            )
            wts_local = np.concatenate(
                [
                    as_boundary_vector(wts, name="RCIP weights")
                    for wts in wtscells
                    if wts is not None
                ]
            )
            local_mat = _eval_kernel(kernel, srcinfo, local_targets)
            vals = vals + local_mat @ _apply_weights_to_density(rho_local, wts_local)

    return as_boundary_field_matrix(
        vals,
        vals.size // targinfo.r.shape[1],
        targinfo.r.shape[1],
        name="RCIP values",
    )


def _merge_pointinfos(infos: list[PointInfo]) -> PointInfo | None:
    if not infos:
        return None
    return PointInfo(
        r=np.column_stack([info.r for info in infos]),
        d=None
        if any(info.d is None for info in infos)
        else np.column_stack([info.d for info in infos]),
        d2=None
        if any(info.d2 is None for info in infos)
        else np.column_stack([info.d2 for info in infos]),
        n=None
        if any(info.n is None for info in infos)
        else np.column_stack([info.n for info in infos]),
        data=None
        if any(info.data is None for info in infos)
        else np.column_stack([info.data for info in infos]),
    )


def _shift_pointinfo(info: PointInfo, center: np.ndarray) -> PointInfo:
    return PointInfo(
        r=info.r - np.asarray(center).reshape(info.r.shape[0], 1),
        d=info.d,
        d2=info.d2,
        n=info.n,
        data=info.data,
    )


def _chunker_from_pointinfo(info: PointInfo, wts: np.ndarray, quadrature_order: int) -> Chunker:
    npt = int(info.r.shape[1])
    quadrature_order = int(quadrature_order)
    if npt % quadrature_order != 0:
        raise ValueError("RCIP local source points must be whole chunks")
    nch = npt // quadrature_order
    out = Chunker(ChunkerPref(k=quadrature_order, dim=info.r.shape[0], nchstor=nch, nchmax=nch))
    out.addchunk(nch)
    out.r = as_boundary_chunk_tensor(
        info.r, info.r.shape[0], quadrature_order, nch, name="positions"
    )
    if info.d is not None:
        out.d = as_boundary_chunk_tensor(
            info.d, info.r.shape[0], quadrature_order, nch, name="derivatives"
        )
    if info.d2 is not None:
        out.d2 = as_boundary_chunk_tensor(
            info.d2, info.r.shape[0], quadrature_order, nch, name="second derivatives"
        )
    if info.n is not None:
        out.n = as_boundary_chunk_tensor(
            info.n, info.r.shape[0], quadrature_order, nch, name="normals"
        )
    out.wts = as_boundary_weight_matrix(wts, quadrature_order, nch, name="weights")
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


def _is_block_kernel_matrix(kernel: Any) -> bool:
    if callable(kernel):
        return False
    try:
        arr = np.asarray(kernel, dtype=object)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 2 and arr.size > 0 and all(callable(item) for item in arr.flat)


def _block_kernel_layout(chunker_collection: Any, kernels: Any) -> _BlockKernelLayout:
    chunkers = _as_chunker_sequence(chunker_collection)
    if chunkers is None:
        raise TypeError(
            "block kernel matrices require a chunker sequence or chunkgraph-like object"
        )
    nchunker = len(chunkers)
    kernel_matrix = np.asarray(kernels, dtype=object)
    if kernel_matrix.shape != (nchunker, nchunker):
        raise ValueError("block kernel matrix shape must match the number of chunkers")

    rowdims = np.zeros(nchunker, dtype=int)
    coldims = np.zeros(nchunker, dtype=int)
    opdims_mat = np.zeros((2, nchunker, nchunker), dtype=int)
    for target_index, target_chunker in enumerate(chunkers):
        for source_index, source_chunker in enumerate(chunkers):
            op0, op1 = _kernel_opdims_between(
                source_chunker,
                target_chunker,
                kernel_matrix[target_index, source_index],
            )
            opdims_mat[:, target_index, source_index] = (op0, op1)
            if rowdims[target_index] == 0:
                rowdims[target_index] = op0
            elif rowdims[target_index] != op0:
                raise ValueError("block kernel row operator dimensions are inconsistent")
            if coldims[source_index] == 0:
                coldims[source_index] = op1
            elif coldims[source_index] != op1:
                raise ValueError("block kernel column operator dimensions are inconsistent")

    row_offsets = np.concatenate(
        (
            [0],
            np.cumsum([chunker.npt * dim for chunker, dim in zip(chunkers, rowdims, strict=True)]),
        )
    )
    col_offsets = np.concatenate(
        (
            [0],
            np.cumsum([chunker.npt * dim for chunker, dim in zip(chunkers, coldims, strict=True)]),
        )
    )
    return _BlockKernelLayout(
        chunkers, kernel_matrix, opdims_mat, rowdims, coldims, row_offsets, col_offsets
    )


def _block_operator_dtype(layout: _BlockKernelLayout) -> np.dtype:
    dtype = np.dtype(float)
    for itarg, targ in enumerate(layout.chunkers):
        for isrc, src in enumerate(layout.chunkers):
            dtype = np.result_type(
                dtype, _operator_dtype_between(src, targ, layout.kernels[itarg, isrc])
            )
    return np.dtype(dtype)


def _block_kernel_mat(chunker_collection: Any, kernels: Any, options: dict[str, Any]) -> np.ndarray:
    layout = _block_kernel_layout(chunker_collection, kernels)
    edge_chunkers = layout.chunkers
    kernel_matrix = layout.kernels
    nedge = len(edge_chunkers)
    out = np.zeros(
        (int(layout.row_offsets[-1]), int(layout.col_offsets[-1])),
        dtype=_block_operator_dtype(layout),
    )

    for kernel in _unique_kernel_objects(kernel_matrix):
        pairs = [
            (target_index, source_index)
            for target_index in range(nedge)
            for source_index in range(nedge)
            if kernel_matrix[target_index, source_index] is kernel
        ]
        target_edges = sorted({target_index for target_index, _ in pairs})
        source_edges = sorted({source_index for _, source_index in pairs})
        targ_merged = merge([edge_chunkers[idx] for idx in target_edges])
        src_merged = merge([edge_chunkers[idx] for idx in source_edges])
        mat = _eval_kernel(kernel, PointInfo.from_any(src_merged), PointInfo.from_any(targ_merged))
        sample_op1 = int(layout.opdims_mat[1, pairs[0][0], pairs[0][1]])
        src_weights = as_boundary_vector(src_merged.wts, name="source weights")
        if mat.shape[1] == src_merged.npt:
            weighted = mat * src_weights[None, :]
        elif mat.shape[1] == src_merged.npt * sample_op1:
            weighted = mat * np.repeat(src_weights, sample_op1)[None, :]
        else:
            raise ValueError(
                "block kernel column dimension is incompatible with source edge points"
            )
        local_rows = _block_offsets_for_edges(edge_chunkers, layout.rowdims, target_edges)
        local_cols = _block_offsets_for_edges(edge_chunkers, layout.coldims, source_edges)
        weighted = _apply_laplace_double_block_self_limits(
            edge_chunkers,
            kernel,
            target_edges,
            source_edges,
            local_rows,
            local_cols,
            weighted,
        )
        target_lookup = {edge: idx for idx, edge in enumerate(target_edges)}
        source_lookup = {edge: idx for idx, edge in enumerate(source_edges)}
        for target_index, source_index in pairs:
            rows = slice(
                int(layout.row_offsets[target_index]), int(layout.row_offsets[target_index + 1])
            )
            cols = slice(
                int(layout.col_offsets[source_index]), int(layout.col_offsets[source_index + 1])
            )
            if target_index == source_index and _uses_special_quadrature(kernel, options):
                local_options = dict(options)
                local_options.pop("l2scale", None)
                out[rows, cols] = chunkermat(edge_chunkers[target_index], kernel, local_options)
                continue
            local_i = target_lookup[target_index]
            local_j = source_lookup[source_index]
            src_rows = slice(int(local_rows[local_i]), int(local_rows[local_i + 1]))
            src_cols = slice(int(local_cols[local_j]), int(local_cols[local_j + 1]))
            out[rows, cols] = weighted[src_rows, src_cols]
    if _l2scale(options):
        return _apply_chunkgraph_l2scale(edge_chunkers, layout.rowdims, layout.coldims, out)
    return out


def _unique_kernel_objects(kernels: np.ndarray) -> list[Callable[[Any, Any], np.ndarray]]:
    unique: list[Callable[[Any, Any], np.ndarray]] = []
    for item in kernels.flat:
        if not any(item is existing for existing in unique):
            unique.append(item)
    return unique


def _block_offsets_for_edges(
    edge_chunkers: list[Chunker], dims: np.ndarray, indices: list[int]
) -> np.ndarray:
    return np.concatenate(
        ([0], np.cumsum([edge_chunkers[idx].npt * int(dims[idx]) for idx in indices]))
    )


def _kernel_opdims_between(
    source: Chunker, target: Chunker, kernel: Callable[[Any, Any], np.ndarray]
) -> tuple[int, int]:
    opdims = getattr(kernel, "opdims", None)
    if opdims is not None and tuple(opdims) != (0, 0):
        return int(opdims[0]), int(opdims[1])
    mat = _eval_kernel(kernel, _pointinfo_node(source, 0), _pointinfo_node(target, 0))
    return int(mat.shape[0]), int(mat.shape[1])


def _operator_dtype_between(
    source: Chunker, target: Chunker, kernel: Callable[[Any, Any], np.ndarray]
) -> np.dtype:
    return _probe_kernel_dtype(kernel, _pointinfo_node(source, 0), _pointinfo_node(target, 0))


def _apply_chunkgraph_l2scale(
    edge_chunkers: list[Chunker], rowdims: np.ndarray, coldims: np.ndarray, mat: np.ndarray
) -> np.ndarray:
    row_scales = np.concatenate(
        [
            boundary_component_weights(np.sqrt(edge.wts), int(dim))
            for edge, dim in zip(edge_chunkers, rowdims, strict=True)
        ]
    )
    col_scales = np.concatenate(
        [
            boundary_component_weights(1.0 / np.sqrt(edge.wts), int(dim))
            for edge, dim in zip(edge_chunkers, coldims, strict=True)
        ]
    )
    if sparse.issparse(mat):
        return sparse.diags(row_scales, format="csr") @ mat @ sparse.diags(col_scales, format="csr")
    return row_scales[:, None] * mat * col_scales[None, :]


def _chunker_l2_row_scale(chunker: Chunker, rowdim: int) -> np.ndarray:
    return boundary_component_weights(np.sqrt(chunker.wts), int(rowdim))


def _chunker_l2_col_scale(chunker: Chunker, coldim: int) -> np.ndarray:
    return boundary_component_weights(1.0 / np.sqrt(chunker.wts), int(coldim))


def _block_l2_row_scale(layout: _BlockKernelLayout) -> np.ndarray:
    return np.concatenate(
        [
            _chunker_l2_row_scale(edge, int(dim))
            for edge, dim in zip(layout.chunkers, layout.rowdims, strict=True)
        ]
    )


def _block_l2_col_scale(layout: _BlockKernelLayout) -> np.ndarray:
    return np.concatenate(
        [
            _chunker_l2_col_scale(edge, int(dim))
            for edge, dim in zip(layout.chunkers, layout.coldims, strict=True)
        ]
    )


def _weighted_density(chunker: Chunker, density: ArrayLike) -> np.ndarray:
    return weighted_density_for_boundary(density, chunker.wts, chunker.npt)


def _density_matmul_arg(ncols: int, density: ArrayLike) -> np.ndarray:
    return density_matmul_argument(density, ncols)


def _eval_kernel(
    kernel: Callable[[Any, Any], np.ndarray], source_info: PointInfo, target_info: PointInfo
) -> np.ndarray:
    if hasattr(kernel, "eval") and kernel.eval is not None:
        return kernel.eval(source_info, target_info)
    return kernel(source_info, target_info)


def _uses_special_quadrature(
    kernel: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any] | _OperatorOptions | None,
) -> bool:
    return _OperatorOptions.from_any(options).uses_special_quadrature(kernel)


def _special_quadrature_type(
    kernel: Callable[[Any, Any], np.ndarray], options: dict[str, Any] | _OperatorOptions
) -> str:
    return _OperatorOptions.from_any(options).special_quadrature_type(kernel)


def _is_laplace_double_kernel(kernel: Callable[[Any, Any], np.ndarray]) -> bool:
    return str(getattr(kernel, "name", "")).lower() == "laplace" and str(
        getattr(kernel, "type", "")
    ).lower() in {
        "d",
        "double",
    }


def _is_laplace_sprime_kernel(kernel: Callable[[Any, Any], np.ndarray]) -> bool:
    return str(getattr(kernel, "name", "")).lower() == "laplace" and str(
        getattr(kernel, "type", "")
    ).lower() in {
        "sp",
        "sprime",
    }


def _is_stokes_strac_kernel(kernel: Callable[[Any, Any], np.ndarray]) -> bool:
    return str(getattr(kernel, "name", "")).lower() in {"stokes", "stok"} and str(
        getattr(kernel, "type", "")
    ).lower() in {
        "strac",
        "straction",
    }


def _apply_laplace_double_self_limit(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    mat: np.ndarray,
) -> np.ndarray:
    if not _is_laplace_double_kernel(kernel) or mat.shape != (chunker.npt, chunker.npt):
        return mat
    scale = getattr(kernel, "params", {}).get("_scale", 1.0)
    speed = np.sqrt(np.sum(chunker.d**2, axis=0))
    curvature = (chunker.d[0] * chunker.d2[1] - chunker.d[1] * chunker.d2[0]) / speed**3
    diag = (
        scale
        * (-as_boundary_vector(curvature, name="curvature") / (4.0 * np.pi))
        * as_boundary_vector(chunker.wts, name="weights")
    )
    np.fill_diagonal(mat, diag)
    return mat


def _apply_laplace_sprime_self_limit(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    mat: np.ndarray,
) -> np.ndarray:
    if not _is_laplace_sprime_kernel(kernel) or mat.shape != (chunker.npt, chunker.npt):
        return mat
    scale = getattr(kernel, "params", {}).get("_scale", 1.0)
    speed = np.sqrt(np.sum(chunker.d**2, axis=0))
    curvature = (chunker.d[0] * chunker.d2[1] - chunker.d[1] * chunker.d2[0]) / speed**3
    diag = (
        scale
        * (-as_boundary_vector(curvature, name="curvature") / (4.0 * np.pi))
        * as_boundary_vector(chunker.wts, name="weights")
    )
    np.fill_diagonal(mat, diag)
    return mat


def _apply_stokes_strac_self_limit(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    mat: np.ndarray,
) -> np.ndarray:
    if not _is_stokes_strac_kernel(kernel) or mat.shape != (2 * chunker.npt, 2 * chunker.npt):
        return mat
    scale = getattr(kernel, "params", {}).get("_scale", 1.0)
    d = as_boundary_point_matrix(chunker.d, chunker.dim, chunker.npt, name="derivatives")
    speed = np.sqrt(np.sum(d**2, axis=0))
    tangents = d / speed[None, :]
    curvature = as_boundary_vector(chunker.signed_curvature(), name="curvature")
    weights = as_boundary_vector(chunker.wts, name="weights")
    for inode in range(chunker.npt):
        block = (
            scale
            * (-curvature[inode] / (2.0 * np.pi))
            * np.outer(tangents[:, inode], tangents[:, inode])
            * weights[inode]
        )
        rows = slice(2 * inode, 2 * inode + 2)
        mat[rows, rows] = block
    return mat


def _apply_laplace_double_block_self_limits(
    edge_chunkers: list[Chunker],
    kernel: Callable[[Any, Any], np.ndarray],
    target_edges: list[int],
    source_edges: list[int],
    target_offsets: np.ndarray,
    source_offsets: np.ndarray,
    mat: np.ndarray,
) -> np.ndarray:
    if not _is_laplace_double_kernel(kernel):
        return mat
    target_lookup = {edge: idx for idx, edge in enumerate(target_edges)}
    source_lookup = {edge: idx for idx, edge in enumerate(source_edges)}
    for edge in sorted(set(target_lookup) & set(source_lookup)):
        i = target_lookup[edge]
        j = source_lookup[edge]
        rows = slice(int(target_offsets[i]), int(target_offsets[i + 1]))
        cols = slice(int(source_offsets[j]), int(source_offsets[j + 1]))
        _apply_laplace_double_self_limit(edge_chunkers[edge], kernel, mat[rows, cols])
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


def _flam_options(
    options: dict[str, Any] | _OperatorOptions, *, store_default: str | None = None
) -> dict[str, Any]:
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


def _require_fmm(kernel: Callable[[Any, Any], np.ndarray]) -> None:
    fmm = getattr(kernel, "fmm", None)
    if fmm is None:
        raise NotImplementedError("FMM acceleration requested, but the kernel has no FMM evaluator")
    if bool(getattr(fmm, "_chunkie_direct_fmm_fallback", False)):
        name = getattr(kernel, "name", "custom")
        typ = getattr(kernel, "type", "")
        label = f"{name} {typ}".strip()
        warnings.warn(
            f"FMM acceleration requested for {label}, but no accelerated FMM evaluator is available; "
            "using a direct dense matrix-vector fallback",
            RuntimeWarning,
            stacklevel=3,
        )


def _require_pyflam():
    try:
        import pyflam
    except (
        ImportError,
        OSError,
    ) as exc:  # pragma: no cover - dependency is required in packaged installs.
        raise ImportError("FLAM acceleration requires the pyflam package") from exc
    return pyflam


def _dval_vector(dval: ArrayLike | float | complex, size: int) -> np.ndarray:
    arr = np.asarray(dval)
    if arr.size == 1:
        return np.full(size, arr.reshape(-1)[0], dtype=arr.dtype)
    vec = as_boundary_vector(arr, name="dval")
    if vec.size != size:
        raise ValueError(f"dval must be scalar or length {size}")
    return vec


def _special_overwrite_matrix(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any],
) -> spmatrix:
    if not _uses_special_quadrature(kernel, options):
        opdims = _kernel_opdims(chunker, kernel)
        return sparse.csr_matrix((chunker.npt * int(opdims[0]), chunker.npt * int(opdims[1])))
    from .quadrature import ggq as quadggq

    qtype = _special_quadrature_type(kernel, options)
    spmat = quadggq.buildmattd(
        chunker,
        kernel,
        getattr(kernel, "opdims", None),
        singularity=qtype,
        ilist=options.get("ilist", None),
        corrections=False,
        pquad_side=_pquad_side(options),
        usepquad=_boundary_pquad_enabled(options),
    )
    if _l2scale(options):
        spmat = _apply_l2scale_matrix(chunker, spmat)
    return spmat


def _block_special_overwrite_matrix(
    layout: _BlockKernelLayout, options: dict[str, Any]
) -> spmatrix:
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    vals_all: list[np.ndarray] = []
    for idx, chunker in enumerate(layout.chunkers):
        kernel = layout.kernels[idx, idx]
        if not _uses_special_quadrature(kernel, options):
            continue
        local_options = dict(options)
        local_options.pop("l2scale", None)
        spmat = _special_overwrite_matrix(chunker, kernel, local_options).tocoo()
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
        return sparse.csr_matrix(
            _apply_chunkgraph_l2scale(layout.chunkers, layout.rowdims, layout.coldims, out)
        )
    return out


def _block_uses_special_quadrature(layout: _BlockKernelLayout, options: dict[str, Any]) -> bool:
    for idx in range(len(layout.chunkers)):
        if _uses_special_quadrature(layout.kernels[idx, idx], options):
            return True
    return False


def _block_special_correction_matrix(
    layout: _BlockKernelLayout, options: dict[str, Any]
) -> spmatrix:
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    vals_all: list[np.ndarray] = []
    for idx, chunker in enumerate(layout.chunkers):
        kernel = layout.kernels[idx, idx]
        if not _uses_special_quadrature(kernel, options):
            continue
        local_options = dict(options)
        local_options.pop("l2scale", None)
        spmat = _special_correction_matrix(chunker, kernel, local_options).tocoo()
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


def _apply_l2scale_matrix(chunker: Chunker, mat: np.ndarray | spmatrix) -> np.ndarray | spmatrix:
    npt = chunker.npt
    if mat.shape[0] % npt != 0 or mat.shape[1] % npt != 0:
        raise ValueError("l2scale matrix dimensions must be multiples of chunker.npt")
    op0 = mat.shape[0] // npt
    op1 = mat.shape[1] // npt
    row_scale = _chunker_l2_row_scale(chunker, op0)
    col_scale = _chunker_l2_col_scale(chunker, op1)
    if sparse.issparse(mat):
        return sparse.diags(row_scale, format="csr") @ mat @ sparse.diags(col_scale, format="csr")
    return row_scale[:, None] * np.asarray(mat) * col_scale[None, :]


def _add_diagonal_shift(
    out: np.ndarray, rows: np.ndarray, cols: np.ndarray, dval: np.ndarray
) -> np.ndarray:
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
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    options: dict[str, Any],
):
    from .acceleration import flam

    rank_or_tol = _flam_rank_or_tol(options)
    optsnpxy = {"rank_or_tol": float(rank_or_tol), "nsrc": _flam_occ(options)}
    width = float(np.max(chunker.max() - chunker.min()))
    proxybylevel = _flag(options, "proxybylevel")
    if not proxybylevel:
        npxy = flam.nproxy_square(kernel, width, optsnpxy)
        if npxy == -1:
            return None
        pr, ptau, pw, pin = flam.proxy_square_pts(npxy)

        def pxyfun(
            x: np.ndarray,
            slf: np.ndarray,
            nbr: np.ndarray,
            box_size: np.ndarray,
            center: np.ndarray,
        ):
            _ = x
            return flam.proxyfun(
                slf,
                nbr,
                box_size,
                center,
                chunker,
                kernel,
                opdims,
                pr,
                ptau,
                pw,
                pin,
                True,
                _l2scale(options),
            )

        return pxyfun

    def pxyfun(
        x: np.ndarray,
        slf: np.ndarray,
        nbr: np.ndarray,
        box_size: np.ndarray,
        center: np.ndarray,
    ):
        _ = x
        level_width = float(np.max(np.asarray(box_size, dtype=float)))
        npxy = flam.nproxy_square(kernel, level_width, optsnpxy)
        if npxy == -1:
            return np.zeros((0, np.asarray(slf).size)), np.asarray(nbr, dtype=np.int64)
        pr, ptau, pw, pin = flam.proxy_square_pts(npxy)
        return flam.proxyfun(
            slf,
            nbr,
            box_size,
            center,
            chunker,
            kernel,
            opdims,
            pr,
            ptau,
            pw,
            pin,
            True,
            _l2scale(options),
        )

    return pxyfun


def _chunkerkerneval_flam(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    density: ArrayLike,
    target: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any],
) -> np.ndarray:
    factor, matfun, out_shape = _chunkerkerneval_flam_factor(chunker, kernel, target, options)
    pyflam = _require_pyflam()
    dens_vec = as_boundary_vector(density, name="density")
    if dens_vec.size != out_shape[1]:
        raise ValueError("density has incompatible size")
    vals = pyflam.ifmm_mv(factor, dens_vec, matfun)
    ntarget = PointInfo.from_any(target).r.shape[1]
    return as_boundary_field_matrix(
        vals,
        np.asarray(vals).size // ntarget,
        ntarget,
        name="FLAM values",
    )


def _chunkerkernevalmat_flam(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    target: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any],
) -> np.ndarray:
    factor, matfun, out_shape = _chunkerkerneval_flam_factor(chunker, kernel, target, options)
    pyflam = _require_pyflam()
    eye = np.eye(out_shape[1], dtype=_operator_dtype(chunker, kernel))
    return np.asarray(pyflam.ifmm_mv(factor, eye, matfun))


def _chunkerkerneval_flam_factor(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    target: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any],
):
    pyflam = _require_pyflam()
    from .acceleration import flam

    targinfo = PointInfo.from_any(target)
    op0, op1 = _kernel_opdims(chunker, kernel, targinfo)
    nrows = targinfo.r.shape[1] * op0
    ncols = chunker.npt * op1

    def matfun(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return flam.kernbyindexr(rows, cols, targinfo, chunker, kernel, (op0, op1))

    rx = np.repeat(np.real(targinfo.r), op0, axis=1)
    cx = np.repeat(np.real(PointInfo.from_any(chunker).r), op1, axis=1)
    occ = _flam_occ(options)
    rank_or_tol = _flam_rank_or_tol(options)
    opts_ifmm = _flam_options(options, store_default="n")
    # Store near and diagonal blocks; application still receives matfun for
    # interactions not retained by the compact representation.
    pxyfun = None
    if _flag(options, "useproxy", True) and chunker.datadim == 0 and targinfo.data is None:
        pxyfun = _chunkerkerneval_proxyfun(chunker, kernel, targinfo, (op0, op1), options)
    factor = pyflam.ifmm(matfun, rx, cx, occ, rank_or_tol, pxyfun, opts_ifmm)
    return factor, matfun, (nrows, ncols)


def _chunkerkerneval_proxyfun(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    targinfo: PointInfo,
    opdims: tuple[int, int],
    options: dict[str, Any],
):
    from .acceleration import flam

    rank_or_tol = _flam_rank_or_tol(options)
    optsnpxy = {"rank_or_tol": float(rank_or_tol), "nsrc": _flam_occ(options)}
    all_points = np.column_stack((np.real(targinfo.r), np.real(PointInfo.from_any(chunker).r)))
    width = float(np.max(np.max(all_points, axis=1) - np.min(all_points, axis=1)))
    proxybylevel = _flag(options, "proxybylevel")
    if not proxybylevel:
        npxy = flam.nproxy_square(kernel, width, optsnpxy)
        if npxy == -1:
            return None
        pr, ptau, pw, pin = flam.proxy_square_pts(npxy)

        def pxyfun(
            rc: str,
            rx: np.ndarray,
            cx: np.ndarray,
            slf: np.ndarray,
            nbr: np.ndarray,
            box_size: np.ndarray,
            center: np.ndarray,
        ):
            return flam.proxyfunr(
                rc,
                rx,
                cx,
                slf,
                nbr,
                box_size,
                center,
                chunker,
                kernel,
                opdims,
                pr,
                ptau,
                pw,
                pin,
                target=targinfo,
            )

        return pxyfun

    def pxyfun(
        rc: str,
        rx: np.ndarray,
        cx: np.ndarray,
        slf: np.ndarray,
        nbr: np.ndarray,
        box_size: np.ndarray,
        center: np.ndarray,
    ):
        level_width = float(np.max(np.asarray(box_size, dtype=float)))
        npxy = flam.nproxy_square(kernel, level_width, optsnpxy)
        if npxy == -1:
            slf_size = np.asarray(slf).size
            if str(rc).lower() == "c":
                return np.zeros((0, slf_size)), np.asarray(nbr, dtype=np.int64)
            return np.zeros((slf_size, 0)), np.asarray(nbr, dtype=np.int64)
        pr, ptau, pw, pin = flam.proxy_square_pts(npxy)
        return flam.proxyfunr(
            rc,
            rx,
            cx,
            slf,
            nbr,
            box_size,
            center,
            chunker,
            kernel,
            opdims,
            pr,
            ptau,
            pw,
            pin,
            target=targinfo,
        )

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
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    density: ArrayLike,
    options: dict[str, Any],
    correction: spmatrix | None = None,
) -> np.ndarray:
    dens_vec = as_boundary_vector(density, name="density")
    op0, op1 = _kernel_opdims(chunker, kernel)
    use_l2scale = _l2scale(options)
    eval_dens = dens_vec * _chunker_l2_col_scale(chunker, op1) if use_l2scale else dens_vec
    special = _uses_special_quadrature(kernel, options)
    fmm_options = _smooth_fmm_options(options) if special else dict(options)
    fmm_options["acceleration"] = "fmm"
    fmm_options.pop("l2scale", None)
    vals = as_boundary_vector(
        chunkerkerneval(chunker, kernel, eval_dens, chunker, fmm_options),
        name="FMM values",
    )
    if special:
        corr = (
            _special_correction_matrix(chunker, kernel, options)
            if correction is None
            else correction
        )
        vals = vals + corr @ eval_dens
    if use_l2scale:
        vals = _chunker_l2_row_scale(chunker, op0) * vals
    return vals


def _block_chunkermatapply_fmm(
    layout: _BlockKernelLayout,
    density: ArrayLike,
    options: dict[str, Any],
    correction: spmatrix | None = None,
) -> np.ndarray:
    dens_vec = as_boundary_vector(density, name="density")
    ncols = int(layout.col_offsets[-1])
    if dens_vec.size != ncols:
        raise ValueError("density has incompatible size")
    use_l2scale = _l2scale(options)
    eval_dens = dens_vec * _block_l2_col_scale(layout) if use_l2scale else dens_vec
    nrows = int(layout.row_offsets[-1])
    out = np.zeros(nrows, dtype=np.result_type(_block_operator_dtype(layout), eval_dens.dtype))
    fmm_options = (
        _smooth_fmm_options(options)
        if _block_uses_special_quadrature(layout, options)
        else dict(options)
    )
    fmm_options["acceleration"] = "fmm"
    fmm_options.pop("l2scale", None)

    for itarg, targ in enumerate(layout.chunkers):
        rows = slice(int(layout.row_offsets[itarg]), int(layout.row_offsets[itarg + 1]))
        for isrc, src in enumerate(layout.chunkers):
            kernel = layout.kernels[itarg, isrc]
            _require_fmm(kernel)
            cols = slice(int(layout.col_offsets[isrc]), int(layout.col_offsets[isrc + 1]))
            vals = as_boundary_vector(
                chunkerkerneval(src, kernel, eval_dens[cols], targ, fmm_options),
                name="block FMM values",
            )
            out[rows] += vals

    if _block_uses_special_quadrature(layout, options):
        corr = (
            _block_special_correction_matrix(layout, options) if correction is None else correction
        )
        out = out + corr @ eval_dens
    if use_l2scale:
        out = _block_l2_row_scale(layout) * out
    return out


def _smooth_fmm_options(options: dict[str, Any]) -> dict[str, Any]:
    out = dict(options)
    out["forcesmooth"] = True
    out.pop("forceadap", None)
    out.pop("adaptive_correction", None)
    return out


def _chunkerkernevalmat_fmm(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    target: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any],
) -> np.ndarray:
    targinfo = PointInfo.from_any(target)
    op0, op1 = _kernel_opdims(chunker, kernel, targinfo)
    nrows = targinfo.r.shape[1] * op0
    ncols = chunker.npt * op1
    eye = np.eye(ncols, dtype=_operator_dtype(chunker, kernel))
    out = np.empty(
        (nrows, ncols),
        dtype=np.result_type(
            eye.dtype, complex if np.iscomplexobj(getattr(kernel, "params", None)) else float
        ),
    )
    fmm_options = dict(options)
    fmm_options["acceleration"] = "fmm"
    for col in range(ncols):
        out[:, col] = as_boundary_vector(
            chunkerkerneval(chunker, kernel, eye[:, col], targinfo, fmm_options),
            name="materialized FMM column",
        )
    return out


def _special_correction_matrix(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any],
) -> spmatrix:
    from .quadrature import ggq as quadggq

    qtype = _special_quadrature_type(kernel, options)
    return quadggq.buildmattd(
        chunker,
        kernel,
        getattr(kernel, "opdims", None),
        singularity=qtype,
        ilist=options.get("ilist", None),
        corrections=True,
        pquad_side=_pquad_side(options),
        usepquad=_boundary_pquad_enabled(options),
    )


def _target_adaptive_correction_matrix(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    targinfo: PointInfo,
    options: dict[str, Any],
) -> spmatrix:
    """Sparse adaptive correction replacing smooth near-target blocks."""

    op0, op1 = _kernel_opdims(chunker, kernel, targinfo)
    ntarget = targinfo.r.shape[1]
    flags = chunker.flagnear(targinfo.r, fac=float(options.get("fac", 1.0)))
    if not np.any(flags):
        return sparse.csr_matrix((op0 * ntarget, op1 * chunker.npt))

    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    vals_all: list[np.ndarray] = []
    for src_chunk in range(chunker.nch):
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
            r=chunker.r[:, :, src_chunk],
            d=chunker.d[:, :, src_chunk],
            d2=chunker.d2[:, :, src_chunk],
            n=chunker.n[:, :, src_chunk],
            data=chunker.data[:, :, src_chunk] if chunker.datadim else None,
        )
        smooth = (
            _eval_kernel(kernel, srcinfo, subinfo)
            * np.repeat(chunker.wts[:, src_chunk], op1)[None, :]
        )
        close_panel = _target_close_panel_matrix(
            chunker, src_chunk, subinfo, kernel, (op0, op1), options
        )
        delta = close_panel - smooth

        global_rows = (target_ids[:, None] * op0 + np.arange(op0)[None, :]).reshape(-1)
        col_start = src_chunk * chunker.k * op1
        global_cols = col_start + np.arange(chunker.k * op1)
        rr, cc = np.indices(delta.shape)
        rows_all.append(global_rows[rr.reshape(-1)])
        cols_all.append(global_cols[cc.reshape(-1)])
        vals_all.append(delta.reshape(-1))

    if not vals_all:
        return sparse.csr_matrix((op0 * ntarget, op1 * chunker.npt))
    return sparse.csr_matrix(
        (np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(op0 * ntarget, op1 * chunker.npt),
    )


def _target_adaptive_matrix(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    targinfo: PointInfo,
    options: dict[str, Any],
) -> np.ndarray:
    """Build a target-evaluation matrix with adaptive close-panel replacements."""

    opdims = _kernel_opdims(chunker, kernel, targinfo)
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    srcinfo = PointInfo.from_any(chunker)
    mat = _eval_kernel(kernel, srcinfo, targinfo)
    wts = as_boundary_vector(chunker.wts, name="weights")
    if mat.shape[1] == chunker.npt:
        mat = mat * wts[None, :]
    elif mat.shape[1] % chunker.npt == 0:
        mat = mat * np.repeat(wts, mat.shape[1] // chunker.npt)[None, :]
    else:
        raise ValueError("kernel column dimension is incompatible with chunker points")

    flags = chunker.flagnear(targinfo.r, fac=float(options.get("fac", 1.0)))
    if not np.any(flags):
        return mat

    for src_chunk in range(chunker.nch):
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
        submat = _target_close_panel_matrix(
            chunker, src_chunk, subinfo, kernel, (op0, op1), options
        )
        col_start = src_chunk * chunker.k * op1
        cols = slice(col_start, col_start + chunker.k * op1)
        for local_idx, target_idx in enumerate(target_ids):
            rows = slice(op0 * target_idx, op0 * (target_idx + 1))
            local_rows = slice(op0 * local_idx, op0 * (local_idx + 1))
            mat[rows, cols] = submat[local_rows, :]
    return mat


def _target_close_panel_matrix(
    chunker: Chunker,
    src_chunk: int,
    targinfo: PointInfo,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    options: dict[str, Any],
) -> np.ndarray:
    from .quadrature import adaptive as quadadap

    pquad_mat, handled = _target_pquad_panel_matrix(
        chunker, src_chunk, targinfo, kernel, opdims, options
    )
    if pquad_mat is not None and np.all(handled):
        return np.real_if_close(pquad_mat)

    adaptive, _, _, iers = quadadap.adapgausswts(
        chunker, src_chunk, targinfo, kernel, opdims, options=options
    )
    warn_iers = iers
    if pquad_mat is not None and np.any(handled):
        op0 = int(opdims[0])
        rows = _target_rows(np.flatnonzero(handled), op0)
        adaptive = np.asarray(adaptive, dtype=np.result_type(adaptive.dtype, pquad_mat.dtype))
        adaptive[rows, :] = pquad_mat[rows, :]
        warn_iers = iers.copy()
        warn_iers[handled] = 0
    quadadap.warn_adaptive_failures(
        warn_iers, src_chunk=src_chunk, context="target adaptive quadrature", stacklevel=3
    )
    return adaptive


def _target_pquad_panel_matrix(
    chunker: Chunker,
    src_chunk: int,
    targinfo: PointInfo,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    options: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray]:
    if not _pquad_enabled(options):
        return None, np.zeros(targinfo.r.shape[1], dtype=bool)
    from .quadrature import panel as pquad

    split_info = pquad.splitinfo_for_kernel(kernel=kernel)
    if split_info is None or tuple(split_info.opdims) != (int(opdims[0]), int(opdims[1])):
        return None, np.zeros(targinfo.r.shape[1], dtype=bool)
    side_tol = options.get("side_tol", None)
    block, handled = pquad.panel_matrix_auto_side(
        chunker=chunker,
        source_chunk=src_chunk,
        target=targinfo,
        split_info=split_info,
        side=_pquad_side(options),
        side_tol=None if side_tol is None else float(side_tol),
    )
    return block, handled


def _kernel_opdims(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    target_info: PointInfo | None = None,
) -> tuple[int, int]:
    opdims = getattr(kernel, "opdims", None)
    if opdims is not None and tuple(opdims) != (0, 0):
        return int(opdims[0]), int(opdims[1])
    source = _pointinfo_node(chunker, 0)
    target = (
        _pointinfo_first(target_info)
        if target_info is not None
        else _pointinfo_node(chunker, 1 if chunker.npt > 1 else 0)
    )
    mat = _eval_kernel(kernel, source, target)
    return int(mat.shape[0]), int(mat.shape[1])


def _operator_dtype(chunker: Chunker, kernel: Callable[[Any, Any], np.ndarray]) -> np.dtype:
    return _probe_kernel_dtype(
        kernel,
        _pointinfo_node(chunker, 0),
        _pointinfo_node(chunker, 1 if chunker.npt > 1 else 0),
    )


def _probe_kernel_dtype(
    kernel: Callable[[Any, Any], np.ndarray], source: PointInfo, target: PointInfo
) -> np.dtype:
    try:
        return np.asarray(_eval_kernel(kernel, source, target)).dtype
    except _KERNEL_PROBE_EXCEPTIONS:
        return np.dtype(float)


def _pointinfo_node(chunker: Chunker, inode: int) -> PointInfo:
    source = PointInfo.from_any(chunker)
    idx = int(inode)
    return _pointinfo_take(source, np.array([idx], dtype=np.int64))


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


def _chunker_polygon_points(chunker: Chunker) -> np.ndarray:
    return _chunker_component_polygons(chunker)[0]


def _chunker_component_polygons(chunker: Chunker, axissym: bool = False) -> list[np.ndarray]:
    sorted_chunker, info = chunker.sort()
    polygons: list[np.ndarray] = []
    start = 0
    for nch, closed in zip(
        np.asarray(info["nchs"], dtype=int),
        np.asarray(info["ifclosed"], dtype=bool),
        strict=True,
    ):
        polygons.append(
            _chunker_component_polygon(sorted_chunker, start, int(nch), bool(closed), axissym)
        )
        start += int(nch)
    if not polygons:
        raise ValueError("chunker has no boundary points")
    return polygons


def _chunker_component_polygon(
    chunker: Chunker, start: int, nch: int, closed: bool, axissym: bool
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    ts = np.linspace(-1.0, 1.0, max(4 * chunker.k, 64))
    interp = lege.matrin(chunker.k, ts)[0]
    for ich in range(start, start + nch):
        panel = (interp @ chunker.r[:, :, ich].T).T
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
    for xa, ya, xb, yb in zip(x0, y0, x1, y1, strict=True):
        crosses = (ya > y) != (yb > y)
        hits = np.zeros_like(crosses)
        if np.any(crosses):
            xhit = (xb - xa) * (y[crosses] - ya) / (yb - ya) + xa
            hits[crosses] = x[crosses] < xhit
        inside ^= hits
    return inside
