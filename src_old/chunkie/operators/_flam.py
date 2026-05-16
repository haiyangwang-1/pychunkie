"""PyFLAM-backed operator construction and target evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

from .._layout import as_boundary_field_matrix, as_boundary_vector
from ..geometry.chunker import Chunker
from ..geometry.pointinfo import PointInfo
from ._blocks import _block_kernel_layout, _is_block_kernel_matrix
from ._common import (
    _add_diagonal_shift,
    _dval_vector,
    _flag,
    _flam_occ,
    _flam_options,
    _flam_rank_or_tol,
    _flamtype,
    _kernel_opdims,
    _l2scale,
    _operator_dtype,
    _require_chunker,
    _require_pyflam,
)
from ._special import _block_special_overwrite_matrix, _special_overwrite_matrix
from .options import _normalize_public_options


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

    from ..acceleration import flam

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

    from ..acceleration import flam

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


def _chunkerflam_proxyfun(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    options: dict[str, Any],
):
    from ..acceleration import flam

    rank_or_tol = _flam_rank_or_tol(options)
    proxy_source_count = _flam_occ(options)
    width = float(np.max(chunker.max() - chunker.min()))
    proxybylevel = _flag(options, "proxybylevel")
    if not proxybylevel:
        npxy = flam.nproxy_square(
            kernel, width, source_count=proxy_source_count, rank_or_tol=float(rank_or_tol)
        )
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
        npxy = flam.nproxy_square(
            kernel, level_width, source_count=proxy_source_count, rank_or_tol=float(rank_or_tol)
        )
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
    from ..acceleration import flam

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
    from ..acceleration import flam

    rank_or_tol = _flam_rank_or_tol(options)
    proxy_source_count = _flam_occ(options)
    all_points = np.column_stack((np.real(targinfo.r), np.real(PointInfo.from_any(chunker).r)))
    width = float(np.max(np.max(all_points, axis=1) - np.min(all_points, axis=1)))
    proxybylevel = _flag(options, "proxybylevel")
    if not proxybylevel:
        npxy = flam.nproxy_square(
            kernel, width, source_count=proxy_source_count, rank_or_tol=float(rank_or_tol)
        )
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
        npxy = flam.nproxy_square(
            kernel, level_width, source_count=proxy_source_count, rank_or_tol=float(rank_or_tol)
        )
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
