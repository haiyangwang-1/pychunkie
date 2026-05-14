"""FLAM callback utilities for chunker-backed kernel matrices.

The functions here mirror MATLAB ``+chnk/+flam`` helpers but use the
0-based NumPy index convention expected by :mod:`pyflam`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix

from .. import lege
from .._layout import as_boundary_tensor, as_boundary_vector
from ..geometry import PointInfo
from ..geometry.chunker import Chunker, merge

_KERNEL_PROBE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    FloatingPointError,
    NotImplementedError,
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


def proxy_square_pts(proxy_order: int = 64, options: dict[str, Any] | None = None):
    """Return square proxy points, tangents, weights, and inside predicate."""

    options = {} if options is None else dict(options)
    proxy_order = int(proxy_order)
    if proxy_order <= 0 or proxy_order % 4 != 0:
        raise ValueError("number of square proxy points must be a positive multiple of 4")
    po4 = proxy_order // 4
    iflege = bool(options.get("iflege", True))

    if not iflege:
        pts = -1.5 + 3.0 * np.arange(po4) / po4
        wts = np.full(po4, 3.0 / po4)
    else:
        k = min(16, po4)
        if po4 % k != 0:
            k = po4
        npanel = po4 // k
        xleg, wleg = lege.exps(k)[:2]
        panels = np.linspace(-1.5, 1.5, npanel + 1)
        pts = np.concatenate(
            [panels[p] + 3.0 / (2.0 * npanel) * (xleg + 1.0) for p in range(npanel)]
        )
        wts = np.tile(3.0 / (2.0 * npanel) * wleg, npanel)

    one = np.ones(po4)
    pr = np.vstack(
        (
            np.concatenate((pts, 1.5 * one, -pts, -1.5 * one)),
            np.concatenate((-1.5 * one, pts, 1.5 * one, -pts)),
        )
    )
    ptau = np.vstack(
        (
            np.concatenate((one, np.zeros(po4), -one, np.zeros(po4))),
            np.concatenate((np.zeros(po4), one, np.zeros(po4), -one)),
        )
    )
    pw = np.tile(wts, 4)

    def pin(x: ArrayLike) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(2, -1)
        return np.max(np.abs(arr), axis=0) < 1.5

    return pr, ptau, pw, pin


def proxy_rect_pts(
    half_lengths: ArrayLike | None = None,
    counts: ArrayLike | None = None,
    options: dict[str, Any] | None = None,
):
    """Return rectangular proxy geometry around ``[-lxy[0],lxy[0]] x [-lxy[1],lxy[1]]``."""

    options = {} if options is None else dict(options)
    half_lengths = (
        np.ones(2) if half_lengths is None else np.asarray(half_lengths, dtype=float).reshape(2)
    )
    counts = (
        np.array([10, 10], dtype=int)
        if counts is None
        else np.asarray(counts, dtype=int).reshape(2)
    )
    if np.any(half_lengths <= 0.0) or np.any(counts <= 0):
        raise ValueError("rectangle proxy lengths and counts must be positive")

    if bool(options.get("iflege", False)):
        pts_x, wts_x = lege.exps(int(counts[0]))[:2]
        pts_y, wts_y = lege.exps(int(counts[1]))[:2]
        pts_x = half_lengths[0] * pts_x
        pts_y = half_lengths[1] * pts_y
        wts_x = half_lengths[0] * wts_x
        wts_y = half_lengths[1] * wts_y
    else:
        pts_x = -half_lengths[0] + 2.0 * half_lengths[0] * np.arange(counts[0]) / counts[0]
        pts_y = -half_lengths[1] + 2.0 * half_lengths[1] * np.arange(counts[1]) / counts[1]
        wts_x = np.full(counts[0], 2.0 * half_lengths[0] / counts[0])
        wts_y = np.full(counts[1], 2.0 * half_lengths[1] / counts[1])

    ox = np.ones_like(pts_x)
    oy = np.ones_like(pts_y)
    pr = np.vstack(
        (
            np.concatenate((pts_x, half_lengths[0] * oy, -pts_x, -half_lengths[0] * oy)),
            np.concatenate((-half_lengths[1] * ox, pts_y, half_lengths[1] * ox, -pts_y)),
        )
    )
    ptau = np.vstack(
        (
            np.concatenate((ox, np.zeros_like(oy), -ox, np.zeros_like(oy))),
            np.concatenate((np.zeros_like(ox), oy, np.zeros_like(ox), -oy)),
        )
    )
    pw = np.concatenate((wts_x, wts_y, wts_x, wts_y))

    def pin(x: ArrayLike) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(2, -1)
        return np.max(np.abs(arr / half_lengths[:, None]), axis=0) < 1.0

    return pr, ptau, pw, pin


def proxy_circ_pts(point_count: int = 64):
    """Return circular proxy points, outward normals, and weights."""

    point_count = int(point_count)
    theta = 2.0 * np.pi * np.arange(point_count) / point_count
    proxy = 1.5 * np.vstack((np.cos(theta), np.sin(theta)))
    pnorm = np.vstack((np.cos(theta), np.sin(theta)))
    pw = np.full(point_count, 2.0 * np.pi * 1.5 / point_count)
    return proxy, pnorm, pw


def nproxy_square(
    kernel: Callable[[Any, Any], np.ndarray],
    width: float,
    options: dict[str, Any] | None = None,
) -> int:
    """Choose a square proxy order by convergence of a random-source test."""

    options = {} if options is None else dict(options)
    nsrc = int(options.get("nsrc", 200))
    rank_or_tol = float(options.get("rank_or_tol", options.get("eps", 1.0e-13)))
    width = float(width)
    if width <= 0.0:
        return 64

    rng = np.random.default_rng(8675309)
    src_d = rng.standard_normal((2, nsrc))
    srcinfo = _new_info(
        r=np.array([[-0.5], [-0.5]]) * width + rng.random((2, nsrc)) * width,
        d=src_d,
        d2=rng.standard_normal((2, nsrc)),
        n=_perp_unit(src_d),
    )
    stmp_d = rng.standard_normal((2, 1))
    stmp = _new_info(
        r=rng.standard_normal((2, 1)),
        d=stmp_d,
        d2=rng.standard_normal((2, 1)),
        n=_perp_unit(stmp_d),
    )
    ttmp_d = rng.standard_normal((2, 1))
    ttmp = _new_info(
        r=rng.standard_normal((2, 1)),
        d=ttmp_d,
        d2=rng.standard_normal((2, 1)),
        n=_perp_unit(ttmp_d),
    )
    op0, op1 = np.asarray(_eval_kernel(kernel, stmp, ttmp)).shape
    sigma = rng.standard_normal(op1 * nsrc)

    npxy = 64
    last = np.nan
    one_more = True
    for _ in range(15):
        pr, ptau, pw, _ = proxy_square_pts(npxy)
        targinfo = _new_info(r=width * pr, d=ptau, d2=np.zeros_like(ptau), n=_perp_unit(ptau))
        pwuse = np.repeat(pw, op0)
        integral = pwuse @ (_eval_kernel(kernel, srcinfo, targinfo) @ sigma)
        denom = max(abs(integral), np.finfo(float).eps)
        err = abs(integral - last) / denom
        if err < rank_or_tol or not one_more:
            return npxy
        npxy *= 2
        last = integral
        if rank_or_tol < 1.0e-12 and err < 1.0e-12:
            one_more = False
    return -1


def proxyfun(
    slf: ArrayLike,
    nbr: ArrayLike,
    box_size: ArrayLike,
    center: ArrayLike,
    chunker: Any,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | ArrayLike | None = None,
    pr: ArrayLike | None = None,
    ptau: ArrayLike | None = None,
    pw: ArrayLike | None = None,
    pin: Callable[[ArrayLike], np.ndarray] | None = None,
    ifaddtrans: bool = True,
    l2scale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Square proxy callback body for ``pyflam.rskelf``.

    ``box_size`` and ``center`` are supplied positionally by pyflam for the
    current compression box. ``slf`` and ``nbr`` are component-interleaved local
    degrees of freedom for the self and neighbor sets.
    """

    boundary = _as_chunker(chunker)
    op0, op1 = _opdims(boundary, kernel, opdims)
    if op0 != op1:
        raise ValueError("square FLAM proxy callbacks require square operator dimensions")
    if pr is None or ptau is None or pw is None or pin is None:
        pr, ptau, pw, pin = proxy_square_pts()

    slf_arr = _as_index_array(slf)
    nbr_arr = _as_index_array(nbr)
    max_box_size = float(np.max(np.asarray(box_size, dtype=float)))
    center_arr = np.asarray(center, dtype=float).reshape(2, 1)
    pinfo = _proxy_info(pr, ptau, max_box_size, center_arr)
    pweights = max_box_size * np.asarray(pw, dtype=float).reshape(-1)
    info = _pointinfo(boundary)
    weights = as_boundary_vector(boundary.wts, name="weights")

    if nbr_arr.size:
        nbr_pts = nbr_arr // op0
        inside = pin((info.r[:, nbr_pts] - center_arr) / max_box_size)
        nbr_arr = nbr_arr[inside]

    proxy_rows = np.arange(pinfo.r.shape[1] * op0, dtype=np.int64)
    Kpxy = _subblock_from_dofs(
        kernel, info, weights, slf_arr, op1, pinfo, pweights, proxy_rows, op0, l2scale=l2scale
    )
    if ifaddtrans:
        proxy_cols = np.arange(pinfo.r.shape[1] * op1, dtype=np.int64)
        Kpxy2 = _subblock_from_dofs(
            kernel,
            pinfo,
            pweights,
            proxy_cols,
            op1,
            info,
            weights,
            slf_arr,
            op0,
            l2scale=l2scale,
        )
        Kpxy = np.vstack((Kpxy, Kpxy2.T))
    return Kpxy, nbr_arr


def proxyfunr(
    rc: str,
    rx: ArrayLike,
    cx: ArrayLike,
    slf: ArrayLike,
    nbr: ArrayLike,
    box_size: ArrayLike,
    center: ArrayLike,
    chunker: Any,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | ArrayLike | None,
    pr: ArrayLike | None = None,
    ptau: ArrayLike | None = None,
    pw: ArrayLike | None = None,
    pin: Callable[[ArrayLike], np.ndarray] | None = None,
    rd: ArrayLike | None = None,
    target: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular proxy callback body for ``pyflam.ifmm``/``rskel``.

    ``rc``, ``rx``, ``cx``, ``slf``, ``nbr``, ``box_size``, and ``center`` are
    pyflam's positional callback parameters. ``target`` may override ``rx``
    when the target geometry is already available as a point-info object.
    """

    _ = cx, rd
    source = _as_chunker(chunker)
    target_info = _pointinfo(rx if target is None else target)
    op0, op1 = _opdims(source, kernel, opdims, target=target_info)
    if pr is None or ptau is None or pw is None or pin is None:
        pr, ptau, pw, pin = proxy_square_pts()

    slf_arr = _as_index_array(slf)
    nbr_arr = _as_index_array(nbr)
    max_box_size = float(np.max(np.asarray(box_size, dtype=float)))
    center_arr = np.asarray(center, dtype=float).reshape(2, 1)
    pinfo = _proxy_info(pr, ptau, max_box_size, center_arr)
    pweights = max_box_size * np.asarray(pw, dtype=float).reshape(-1)
    source_info = _pointinfo(source)
    source_weights = as_boundary_vector(source.wts, name="source weights")

    if str(rc).lower() == "c":
        proxy_rows = np.arange(pinfo.r.shape[1] * op0, dtype=np.int64)
        Kpxy = _subblock_from_dofs(
            kernel,
            source_info,
            source_weights,
            slf_arr,
            op1,
            pinfo,
            pweights,
            proxy_rows,
            op0,
        )
        if nbr_arr.size:
            nbr_pts = nbr_arr // op0
            inside = pin((target_info.r[:, nbr_pts] - center_arr) / max_box_size)
            nbr_arr = nbr_arr[inside]
        return Kpxy, nbr_arr

    proxy_cols = np.arange(pinfo.r.shape[1] * op1, dtype=np.int64)
    Kpxy = _subblock_from_dofs(
        kernel,
        pinfo,
        pweights,
        proxy_cols,
        op1,
        target_info,
        np.ones(target_info.r.shape[1]),
        slf_arr,
        op0,
    )
    if nbr_arr.size:
        nbr_pts = nbr_arr // op1
        inside = pin((source_info.r[:, nbr_pts] - center_arr) / max_box_size)
        nbr_arr = nbr_arr[inside]
    return Kpxy, nbr_arr


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


def _as_chunker(obj: Any) -> Chunker:
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
    raise TypeError("expected a chunker or chunkgraph-like object")


def _chunker_sequence(obj: Any) -> list[Chunker]:
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
        items = list(edges)
        if items and all(isinstance(item, Chunker) for item in items):
            return items
    raise TypeError("expected a chunker sequence or chunkgraph-like object")


def _is_block_kernel_matrix(kernel: Any) -> bool:
    if callable(kernel):
        return False
    try:
        arr = np.asarray(kernel, dtype=object)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 2 and arr.size > 0 and all(callable(item) for item in arr.flat)


def _pointinfo(obj: Any) -> Any:
    return PointInfo.from_any(obj)


def _proxy_info(pr: ArrayLike, ptau: ArrayLike, scale: float, center: np.ndarray) -> Any:
    pxy = np.asarray(pr, dtype=float).reshape(2, -1) * scale + center
    tau = np.asarray(ptau, dtype=float).reshape(2, -1)
    return _new_info(r=pxy, d=tau, d2=np.zeros_like(tau), n=_perp_unit(tau))


def _subset_info(info: Any, indices: np.ndarray) -> Any:
    return _new_info(
        r=info.r[:, indices],
        d=None if info.d is None else info.d[:, indices],
        d2=None if info.d2 is None else info.d2[:, indices],
        n=None if info.n is None else info.n[:, indices],
        data=None if info.data is None else info.data[:, indices],
    )


def _opdims(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | ArrayLike | None,
    *,
    target: Any | None = None,
) -> tuple[int, int]:
    if opdims is not None:
        arr = np.asarray(opdims, dtype=int).reshape(-1)
        if arr.size < 2:
            raise ValueError("opdims must contain two entries")
        return int(arr[0]), int(arr[1])
    kernel_dims = getattr(kernel, "opdims", None)
    if kernel_dims is not None and tuple(kernel_dims) != (0, 0):
        return int(kernel_dims[0]), int(kernel_dims[1])
    source = _subset_info(_pointinfo(chunker), np.array([0], dtype=np.int64))
    target_info = _pointinfo(chunker if target is None else target)
    target_subset = _subset_info(target_info, np.array([0], dtype=np.int64))
    mat = np.asarray(_eval_kernel(kernel, source, target_subset))
    return int(mat.shape[0]), int(mat.shape[1])


def _eval_kernel(
    kernel: Callable[[Any, Any], np.ndarray], source_info: Any, target_info: Any
) -> np.ndarray:
    try:
        eval_method = kernel.eval
    except AttributeError:
        eval_method = None
    if eval_method is not None:
        return eval_method(source_info, target_info)
    return kernel(source_info, target_info)


def _overwrite_sparse(
    out: np.ndarray, rows: np.ndarray, cols: np.ndarray, spmat: spmatrix | None
) -> np.ndarray:
    if spmat is None or rows.size == 0 or cols.size == 0:
        return out
    sub = spmat[np.ix_(rows, cols)].tocoo()
    if sub.nnz:
        out = np.array(out, copy=True)
        out[sub.row, sub.col] = sub.data
    return out


def _as_index_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _validate_points(points: np.ndarray, npt: int, label: str) -> None:
    if np.any(points < 0) or np.any(points >= npt):
        raise IndexError(f"{label} FLAM index is out of range")


def _perp_unit(vec: np.ndarray) -> np.ndarray:
    speed = np.sqrt(np.sum(np.abs(vec) ** 2, axis=0))
    return np.vstack((-vec[1], vec[0])) / speed[None, :]


def _new_info(
    r: np.ndarray,
    d: np.ndarray | None = None,
    d2: np.ndarray | None = None,
    n: np.ndarray | None = None,
    data: np.ndarray | None = None,
) -> Any:
    return PointInfo(r=r, d=d, d2=d2, n=n, data=data)


__all__ = [
    "kernbyindex",
    "kernbyindexr",
    "nproxy_square",
    "proxy_circ_pts",
    "proxy_rect_pts",
    "proxy_square_pts",
    "proxyfun",
    "proxyfunr",
]
