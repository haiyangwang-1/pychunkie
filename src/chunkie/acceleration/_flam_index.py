"""FLAM index callback helpers for boundary matrices."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix

from .._layout import as_boundary_tensor, as_boundary_vector
from ._flam_common import (
    _KERNEL_PROBE_EXCEPTIONS,
    _as_chunker,
    _as_index_array,
    _chunker_sequence,
    _eval_kernel,
    _is_block_kernel_matrix,
    _opdims,
    _overwrite_sparse,
    _pointinfo,
    _subset_info,
    _validate_points,
)


def kernbyindex(
    rows: ArrayLike,
    cols: ArrayLike,
    chunker: Any,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | ArrayLike | None = None,
    spmat: spmatrix | None = None,
    l2scale: bool = False,
) -> np.ndarray:
    """Return weighted boundary operator entries for FLAM index callbacks."""

    if _is_block_kernel_matrix(kernel):
        return _block_kernbyindex(rows, cols, chunker, kernel, opdims, spmat, l2scale)

    boundary = _as_chunker(chunker)
    row_ids = _as_index_array(rows)
    col_ids = _as_index_array(cols)
    op0, op1 = _opdims(boundary, kernel, opdims)
    info = _pointinfo(boundary)
    weights = as_boundary_vector(boundary.wts, name="weights")
    out = _subblock_from_dofs(
        kernel,
        info,
        weights,
        col_ids,
        op1,
        info,
        weights,
        row_ids,
        op0,
        l2scale=l2scale,
    )
    return _overwrite_sparse(out, row_ids, col_ids, spmat)


def kernbyindexr(
    rows: ArrayLike,
    cols: ArrayLike,
    target: Any,
    chunker: Any,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | ArrayLike | None = None,
    spmat: spmatrix | None = None,
) -> np.ndarray:
    """Return weighted target/source operator entries for rectangular FLAM."""

    source = _as_chunker(chunker)
    row_ids = _as_index_array(rows)
    col_ids = _as_index_array(cols)
    source_info = _pointinfo(source)
    target_info = _pointinfo(target)
    op0, op1 = _opdims(source, kernel, opdims, target=target_info)
    src_weights = as_boundary_vector(source.wts, name="source weights")
    target_weights = np.ones(target_info.r.shape[1])
    out = _subblock_from_dofs(
        kernel,
        source_info,
        src_weights,
        col_ids,
        op1,
        target_info,
        target_weights,
        row_ids,
        op0,
    )
    return _overwrite_sparse(out, row_ids, col_ids, spmat)


def _subblock_from_dofs(
    kernel: Callable[[Any, Any], np.ndarray],
    source_info: Any,
    source_weights: np.ndarray,
    source_dofs: np.ndarray,
    source_opdim: int,
    target_info: Any,
    target_weights: np.ndarray,
    target_dofs: np.ndarray,
    target_opdim: int,
    *,
    l2scale: bool = False,
) -> np.ndarray:
    source_dofs = _as_index_array(source_dofs)
    target_dofs = _as_index_array(target_dofs)
    if source_dofs.size == 0 or target_dofs.size == 0:
        return np.zeros((target_dofs.size, source_dofs.size))

    source_points = source_dofs // source_opdim
    source_components = source_dofs % source_opdim
    target_points = target_dofs // target_opdim
    target_components = target_dofs % target_opdim
    _validate_points(source_points, source_info.r.shape[1], "source")
    _validate_points(target_points, target_info.r.shape[1], "target")
    source_unique, source_inverse = np.unique(source_points, return_inverse=True)
    target_unique, target_inverse = np.unique(target_points, return_inverse=True)

    mat = np.asarray(
        _eval_kernel(
            kernel,
            _subset_info(source_info, source_unique),
            _subset_info(target_info, target_unique),
        )
    )
    source_w = np.asarray(source_weights, dtype=float).reshape(-1)[source_unique]
    target_w = np.asarray(target_weights, dtype=float).reshape(-1)[target_unique]
    if l2scale:
        mat = (
            np.sqrt(np.repeat(target_w, target_opdim))[:, None]
            * mat
            * np.sqrt(np.repeat(source_w, source_opdim))[None, :]
        )
    else:
        mat = mat * np.repeat(source_w, source_opdim)[None, :]

    row_lookup = target_inverse * target_opdim + target_components
    col_lookup = source_inverse * source_opdim + source_components
    return mat[np.ix_(row_lookup, col_lookup)]


def _block_kernbyindex(
    rows: ArrayLike,
    cols: ArrayLike,
    chunker_collection: Any,
    kernels: Any,
    opdims: tuple[int, int] | ArrayLike | None = None,
    spmat: spmatrix | None = None,
    l2scale: bool = False,
) -> np.ndarray:
    row_ids = _as_index_array(rows)
    col_ids = _as_index_array(cols)
    layout = _block_layout(chunker_collection, kernels, opdims)
    if row_ids.size == 0 or col_ids.size == 0:
        return np.zeros((row_ids.size, col_ids.size))
    if np.any(row_ids < 0) or np.any(row_ids >= layout["row_offsets"][-1]):
        raise IndexError("target FLAM index is out of range")
    if np.any(col_ids < 0) or np.any(col_ids >= layout["col_offsets"][-1]):
        raise IndexError("source FLAM index is out of range")

    row_chunk = np.searchsorted(layout["row_offsets"][1:], row_ids, side="right")
    col_chunk = np.searchsorted(layout["col_offsets"][1:], col_ids, side="right")
    out = np.zeros((row_ids.size, col_ids.size), dtype=_block_dtype(layout))
    infos = [_pointinfo(chunker) for chunker in layout["chunkers"]]
    weights = [as_boundary_vector(chunker.wts, name="weights") for chunker in layout["chunkers"]]

    for target_index in range(len(layout["chunkers"])):
        row_pos = np.flatnonzero(row_chunk == target_index)
        if row_pos.size == 0:
            continue
        local_rows = row_ids[row_pos] - layout["row_offsets"][target_index]
        row_dim = int(layout["rowdims"][target_index])
        target_dofs = (local_rows // row_dim) * row_dim + (local_rows % row_dim)
        for source_index in range(len(layout["chunkers"])):
            col_pos = np.flatnonzero(col_chunk == source_index)
            if col_pos.size == 0:
                continue
            local_cols = col_ids[col_pos] - layout["col_offsets"][source_index]
            col_dim = int(layout["coldims"][source_index])
            source_dofs = (local_cols // col_dim) * col_dim + (local_cols % col_dim)
            block = _subblock_from_dofs(
                layout["kernels"][target_index, source_index],
                infos[source_index],
                weights[source_index],
                source_dofs,
                col_dim,
                infos[target_index],
                weights[target_index],
                target_dofs,
                row_dim,
                l2scale=l2scale,
            )
            out[np.ix_(row_pos, col_pos)] = block
    return _overwrite_sparse(out, row_ids, col_ids, spmat)


def _block_layout(
    chunker_collection: Any, kernels: Any, opdims: tuple[int, int] | ArrayLike | None
) -> dict[str, Any]:
    chunkers = _chunker_sequence(chunker_collection)
    kernels = np.asarray(kernels, dtype=object)
    nchunker = len(chunkers)
    if kernels.shape != (nchunker, nchunker):
        raise ValueError("block kernel matrix shape must match the number of chunkers")

    rowdims = np.zeros(nchunker, dtype=int)
    coldims = np.zeros(nchunker, dtype=int)
    if opdims is not None:
        arr = np.asarray(opdims, dtype=int)
        if arr.shape == (2, nchunker, nchunker):
            opdims_mat = arr
        else:
            flat = arr.reshape(-1)
            if flat.size != 2 * nchunker * nchunker:
                raise ValueError("block opdims must have shape (2, nchunker, nchunker)")
            opdims_mat = as_boundary_tensor(flat, (2, nchunker, nchunker), name="block opdims")
    else:
        opdims_mat = np.zeros((2, nchunker, nchunker), dtype=int)
        for target_index, target_chunker in enumerate(chunkers):
            for source_index, source_chunker in enumerate(chunkers):
                op0, op1 = _opdims(
                    source_chunker,
                    kernels[target_index, source_index],
                    None,
                    target=target_chunker,
                )
                opdims_mat[:, target_index, source_index] = (op0, op1)

    for target_index in range(nchunker):
        for source_index in range(nchunker):
            op0 = int(opdims_mat[0, target_index, source_index])
            op1 = int(opdims_mat[1, target_index, source_index])
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
    return {
        "chunkers": chunkers,
        "kernels": kernels,
        "opdims_mat": opdims_mat,
        "rowdims": rowdims,
        "coldims": coldims,
        "row_offsets": row_offsets,
        "col_offsets": col_offsets,
    }


def _block_dtype(layout: dict[str, Any]) -> np.dtype:
    dtype = np.dtype(float)
    for target_index, target_chunker in enumerate(layout["chunkers"]):
        target_info = _subset_info(_pointinfo(target_chunker), np.array([0], dtype=np.int64))
        for source_index, source_chunker in enumerate(layout["chunkers"]):
            source_info = _subset_info(_pointinfo(source_chunker), np.array([0], dtype=np.int64))
            try:
                kernel_dtype = np.asarray(
                    _eval_kernel(
                        layout["kernels"][target_index, source_index],
                        source_info,
                        target_info,
                    )
                ).dtype
                dtype = np.result_type(dtype, kernel_dtype)
            except _KERNEL_PROBE_EXCEPTIONS:
                pass
    return np.dtype(dtype)
