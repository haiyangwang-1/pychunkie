"""Special quadrature and close-target operator matrices."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import spmatrix

from .._layout import as_boundary_vector
from ..geometry.chunker import Chunker
from ..geometry.pointinfo import PointInfo
from ._blocks import _apply_chunkgraph_l2scale, _apply_l2scale_matrix
from ._common import (
    _boundary_pquad_enabled,
    _eval_kernel,
    _kernel_opdims,
    _l2scale,
    _pquad_enabled,
    _pquad_side,
    _special_quadrature_type,
    _target_rows,
    _uses_special_quadrature,
)
from .types import _BlockKernelLayout


def _special_overwrite_matrix(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any],
) -> spmatrix:
    if not _uses_special_quadrature(kernel, options):
        opdims = _kernel_opdims(chunker, kernel)
        return sparse.csr_matrix((chunker.npt * int(opdims[0]), chunker.npt * int(opdims[1])))
    from ..quadrature import ggq as quadggq

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


def _special_correction_matrix(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any],
) -> spmatrix:
    from ..quadrature import ggq as quadggq

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
    from ..quadrature import adaptive as quadadap

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
    from ..quadrature import panel as pquad

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
