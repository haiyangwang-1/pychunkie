"""Public operator assembly and matrix-application entry points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .._layout import as_boundary_point_matrix, as_boundary_vector
from ..geometry.chunker import Chunker, merge
from ..geometry.pointinfo import PointInfo
from ._blocks import (
    _apply_chunkgraph_l2scale,
    _apply_l2scale_matrix,
    _apply_laplace_double_block_self_limits,
    _block_kernel_layout,
    _block_offsets_for_edges,
    _block_operator_dtype,
    _is_block_kernel_matrix,
    _unique_kernel_objects,
)
from ._common import (
    _acceleration,
    _apply_laplace_double_self_limit,
    _apply_laplace_sprime_self_limit,
    _apply_stokes_strac_self_limit,
    _boundary_pquad_enabled,
    _density_matmul_arg,
    _eval_kernel,
    _flag,
    _is_chunkgraph_like,
    _l2scale,
    _pquad_side,
    _require_chunker,
    _require_fmm,
    _special_quadrature_type,
    _uses_special_quadrature,
)
from ._matrices import ChunkerFLAMMatrix, ChunkerFMMMatrix
from ._rcip import _chunkgraph_rcip_mat, _chunkgraph_rcip_mat_enabled, _rcip_option_enabled
from .options import _normalize_public_options
from .types import ChunkerRCIPMatrix, RCIPContext


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
        mat, context = _chunkgraph_rcip_mat(chunker, kernel, operator_options, chunkermat)
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
            from ..quadrature import adaptive as quadadap

            adap_options = dict(operator_options)
            adap_options.setdefault("sing", _special_quadrature_type(kernel, operator_options))
            adap_options.setdefault("usepquad", _boundary_pquad_enabled(operator_options))
            mat = quadadap.buildmat(boundary, kernel, getattr(kernel, "opdims", None), adap_options)
        else:
            from ..quadrature import ggq as quadggq

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
