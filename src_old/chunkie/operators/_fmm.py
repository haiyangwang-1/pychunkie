"""FMM-backed operator application helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix

from .._layout import as_boundary_vector
from ..geometry.chunker import Chunker
from ..geometry.pointinfo import PointInfo
from ._blocks import (
    _block_l2_col_scale,
    _block_l2_row_scale,
    _block_operator_dtype,
    _chunker_l2_col_scale,
    _chunker_l2_row_scale,
)
from ._common import (
    _kernel_opdims,
    _l2scale,
    _operator_dtype,
    _require_fmm,
    _uses_special_quadrature,
)
from ._special import (
    _block_special_correction_matrix,
    _block_uses_special_quadrature,
    _special_correction_matrix,
)
from .types import _BlockKernelLayout


def _chunkermatapply_fmm(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    density: ArrayLike,
    options: dict[str, Any],
    correction: spmatrix | None = None,
) -> np.ndarray:
    from ._evaluation import chunkerkerneval

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
    from ._evaluation import chunkerkerneval

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
    from ._evaluation import chunkerkerneval

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
