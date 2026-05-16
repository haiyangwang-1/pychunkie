"""Block-kernel layout and scaling helpers for operators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import spmatrix

from .._layout import boundary_component_weights
from ..geometry.chunker import Chunker
from ._common import (
    _apply_laplace_double_self_limit,
    _as_chunker_sequence,
    _eval_kernel,
    _is_laplace_double_kernel,
    _pointinfo_node,
    _probe_kernel_dtype,
)
from .types import _BlockKernelLayout


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
