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
from ..chunker import Chunker, merge


def kernbyindex(
    i: ArrayLike,
    j: ArrayLike,
    chnkobj: Any,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | ArrayLike | None = None,
    spmat: spmatrix | None = None,
    l2scale: bool = False,
) -> np.ndarray:
    """Return weighted boundary operator entries for FLAM index callbacks."""

    if _is_block_kernel_matrix(kern):
        return _block_kernbyindex(i, j, chnkobj, kern, opdims, spmat, l2scale)

    chnkr = _as_chunker(chnkobj)
    rows = _as_index_array(i)
    cols = _as_index_array(j)
    op0, op1 = _opdims(chnkr, kern, opdims)
    info = _pointinfo(chnkr)
    weights = chnkr.wts.reshape(-1, order="F")
    out = _subblock_from_dofs(
        kern,
        info,
        weights,
        cols,
        op1,
        info,
        weights,
        rows,
        op0,
        l2scale=l2scale,
    )
    return _overwrite_sparse(out, rows, cols, spmat)


def kernbyindexr(
    i: ArrayLike,
    j: ArrayLike,
    targobj: Any,
    chnkr: Any,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | ArrayLike | None = None,
    spmat: spmatrix | None = None,
) -> np.ndarray:
    """Return weighted target/source operator entries for rectangular FLAM."""

    src = _as_chunker(chnkr)
    rows = _as_index_array(i)
    cols = _as_index_array(j)
    op0, op1 = _opdims(src, kern, opdims, targobj=targobj)
    srcinfo = _pointinfo(src)
    targinfo = _pointinfo(targobj)
    src_weights = src.wts.reshape(-1, order="F")
    targ_weights = np.ones(targinfo.r.shape[1])
    out = _subblock_from_dofs(
        kern,
        srcinfo,
        src_weights,
        cols,
        op1,
        targinfo,
        targ_weights,
        rows,
        op0,
    )
    return _overwrite_sparse(out, rows, cols, spmat)


def proxy_square_pts(porder: int = 64, opts: dict[str, Any] | None = None):
    """Return square proxy points, tangents, weights, and inside predicate."""

    options = {} if opts is None else dict(opts)
    porder = int(porder)
    if porder <= 0 or porder % 4 != 0:
        raise ValueError("number of square proxy points must be a positive multiple of 4")
    po4 = porder // 4
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
        pts = np.concatenate([panels[p] + 3.0 / (2.0 * npanel) * (xleg + 1.0) for p in range(npanel)])
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


def proxy_rect_pts(lxy: ArrayLike | None = None, npxy: ArrayLike | None = None, opts: dict[str, Any] | None = None):
    """Return rectangular proxy geometry around ``[-lxy[0],lxy[0]] x [-lxy[1],lxy[1]]``."""

    options = {} if opts is None else dict(opts)
    half_lengths = np.ones(2) if lxy is None else np.asarray(lxy, dtype=float).reshape(2)
    counts = np.array([10, 10], dtype=int) if npxy is None else np.asarray(npxy, dtype=int).reshape(2)
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


def proxy_circ_pts(p: int = 64):
    """Return circular proxy points, outward normals, and weights."""

    p = int(p)
    theta = 2.0 * np.pi * np.arange(p) / p
    proxy = 1.5 * np.vstack((np.cos(theta), np.sin(theta)))
    pnorm = np.vstack((np.cos(theta), np.sin(theta)))
    pw = np.full(p, 2.0 * np.pi * 1.5 / p)
    return proxy, pnorm, pw


def nproxy_square(kern: Callable[[Any, Any], np.ndarray], width: float, opts: dict[str, Any] | None = None) -> int:
    """Choose a square proxy order by convergence of a random-source test."""

    options = {} if opts is None else dict(opts)
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
    stmp = _new_info(r=rng.standard_normal((2, 1)), d=stmp_d, d2=rng.standard_normal((2, 1)), n=_perp_unit(stmp_d))
    ttmp_d = rng.standard_normal((2, 1))
    ttmp = _new_info(r=rng.standard_normal((2, 1)), d=ttmp_d, d2=rng.standard_normal((2, 1)), n=_perp_unit(ttmp_d))
    op0, op1 = np.asarray(_eval_kernel(kern, stmp, ttmp)).shape
    sigma = rng.standard_normal(op1 * nsrc)

    npxy = 64
    last = np.nan
    one_more = True
    for _ in range(15):
        pr, ptau, pw, _ = proxy_square_pts(npxy)
        targinfo = _new_info(r=width * pr, d=ptau, d2=np.zeros_like(ptau), n=_perp_unit(ptau))
        pwuse = np.repeat(pw, op0)
        integral = pwuse @ (_eval_kernel(kern, srcinfo, targinfo) @ sigma)
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
    l: ArrayLike,
    ctr: ArrayLike,
    chnkobj: Any,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | ArrayLike | None = None,
    pr: ArrayLike | None = None,
    ptau: ArrayLike | None = None,
    pw: ArrayLike | None = None,
    pin: Callable[[ArrayLike], np.ndarray] | None = None,
    ifaddtrans: bool = True,
    l2scale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Square proxy callback body for ``pyflam.rskelf``."""

    chnkr = _as_chunker(chnkobj)
    op0, op1 = _opdims(chnkr, kern, opdims)
    if op0 != op1:
        raise ValueError("square FLAM proxy callbacks require square operator dimensions")
    if pr is None or ptau is None or pw is None or pin is None:
        pr, ptau, pw, pin = proxy_square_pts()

    slf_arr = _as_index_array(slf)
    nbr_arr = _as_index_array(nbr)
    lmax = float(np.max(np.asarray(l, dtype=float)))
    ctr_arr = np.asarray(ctr, dtype=float).reshape(2, 1)
    pinfo = _proxy_info(pr, ptau, lmax, ctr_arr)
    pweights = lmax * np.asarray(pw, dtype=float).reshape(-1)
    info = _pointinfo(chnkr)
    weights = chnkr.wts.reshape(-1, order="F")

    if nbr_arr.size:
        nbr_pts = nbr_arr // op0
        inside = pin((info.r[:, nbr_pts] - ctr_arr) / lmax)
        nbr_arr = nbr_arr[inside]

    proxy_rows = np.arange(pinfo.r.shape[1] * op0, dtype=np.int64)
    Kpxy = _subblock_from_dofs(kern, info, weights, slf_arr, op1, pinfo, pweights, proxy_rows, op0, l2scale=l2scale)
    if ifaddtrans:
        proxy_cols = np.arange(pinfo.r.shape[1] * op1, dtype=np.int64)
        Kpxy2 = _subblock_from_dofs(kern, pinfo, pweights, proxy_cols, op1, info, weights, slf_arr, op0, l2scale=l2scale)
        Kpxy = np.vstack((Kpxy, Kpxy2.T))
    return Kpxy, nbr_arr


def proxyfunr(
    rc: str,
    rx: ArrayLike,
    cx: ArrayLike,
    slf: ArrayLike,
    nbr: ArrayLike,
    l: ArrayLike,
    ctr: ArrayLike,
    chnkr: Any,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | ArrayLike | None,
    pr: ArrayLike | None = None,
    ptau: ArrayLike | None = None,
    pw: ArrayLike | None = None,
    pin: Callable[[ArrayLike], np.ndarray] | None = None,
    rd: ArrayLike | None = None,
    targobj: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular proxy callback body for ``pyflam.ifmm``/``rskel``."""

    _ = cx, rd
    src = _as_chunker(chnkr)
    target = _pointinfo(rx if targobj is None else targobj)
    op0, op1 = _opdims(src, kern, opdims, targobj=target)
    if pr is None or ptau is None or pw is None or pin is None:
        pr, ptau, pw, pin = proxy_square_pts()

    slf_arr = _as_index_array(slf)
    nbr_arr = _as_index_array(nbr)
    lmax = float(np.max(np.asarray(l, dtype=float)))
    ctr_arr = np.asarray(ctr, dtype=float).reshape(2, 1)
    pinfo = _proxy_info(pr, ptau, lmax, ctr_arr)
    pweights = lmax * np.asarray(pw, dtype=float).reshape(-1)
    srcinfo = _pointinfo(src)
    src_weights = src.wts.reshape(-1, order="F")

    if str(rc).lower() == "c":
        proxy_rows = np.arange(pinfo.r.shape[1] * op0, dtype=np.int64)
        Kpxy = _subblock_from_dofs(kern, srcinfo, src_weights, slf_arr, op1, pinfo, pweights, proxy_rows, op0)
        if nbr_arr.size:
            nbr_pts = nbr_arr // op0
            inside = pin((target.r[:, nbr_pts] - ctr_arr) / lmax)
            nbr_arr = nbr_arr[inside]
        return Kpxy, nbr_arr

    proxy_cols = np.arange(pinfo.r.shape[1] * op1, dtype=np.int64)
    Kpxy = _subblock_from_dofs(kern, pinfo, pweights, proxy_cols, op1, target, np.ones(target.r.shape[1]), slf_arr, op0)
    if nbr_arr.size:
        nbr_pts = nbr_arr // op1
        inside = pin((srcinfo.r[:, nbr_pts] - ctr_arr) / lmax)
        nbr_arr = nbr_arr[inside]
    return Kpxy, nbr_arr


def _subblock_from_dofs(
    kern: Callable[[Any, Any], np.ndarray],
    srcinfo: Any,
    src_weights: np.ndarray,
    src_dofs: np.ndarray,
    src_opdim: int,
    targinfo: Any,
    targ_weights: np.ndarray,
    targ_dofs: np.ndarray,
    targ_opdim: int,
    *,
    l2scale: bool = False,
) -> np.ndarray:
    src_dofs = _as_index_array(src_dofs)
    targ_dofs = _as_index_array(targ_dofs)
    if src_dofs.size == 0 or targ_dofs.size == 0:
        return np.zeros((targ_dofs.size, src_dofs.size))

    src_pts = src_dofs // src_opdim
    src_comp = src_dofs % src_opdim
    targ_pts = targ_dofs // targ_opdim
    targ_comp = targ_dofs % targ_opdim
    _validate_points(src_pts, srcinfo.r.shape[1], "source")
    _validate_points(targ_pts, targinfo.r.shape[1], "target")
    src_unique, src_inv = np.unique(src_pts, return_inverse=True)
    targ_unique, targ_inv = np.unique(targ_pts, return_inverse=True)

    mat = np.asarray(
        _eval_kernel(kern, _subset_info(srcinfo, src_unique), _subset_info(targinfo, targ_unique))
    )
    src_w = np.asarray(src_weights, dtype=float).reshape(-1)[src_unique]
    targ_w = np.asarray(targ_weights, dtype=float).reshape(-1)[targ_unique]
    if l2scale:
        mat = np.sqrt(np.repeat(targ_w, targ_opdim))[:, None] * mat * np.sqrt(np.repeat(src_w, src_opdim))[None, :]
    else:
        mat = mat * np.repeat(src_w, src_opdim)[None, :]

    row_lookup = targ_inv * targ_opdim + targ_comp
    col_lookup = src_inv * src_opdim + src_comp
    return mat[np.ix_(row_lookup, col_lookup)]


def _block_kernbyindex(
    i: ArrayLike,
    j: ArrayLike,
    chnkobj: Any,
    kerns: Any,
    opdims: tuple[int, int] | ArrayLike | None = None,
    spmat: spmatrix | None = None,
    l2scale: bool = False,
) -> np.ndarray:
    rows = _as_index_array(i)
    cols = _as_index_array(j)
    layout = _block_layout(chnkobj, kerns, opdims)
    if rows.size == 0 or cols.size == 0:
        return np.zeros((rows.size, cols.size))
    if np.any(rows < 0) or np.any(rows >= layout["row_offsets"][-1]):
        raise IndexError("target FLAM index is out of range")
    if np.any(cols < 0) or np.any(cols >= layout["col_offsets"][-1]):
        raise IndexError("source FLAM index is out of range")

    row_chunk = np.searchsorted(layout["row_offsets"][1:], rows, side="right")
    col_chunk = np.searchsorted(layout["col_offsets"][1:], cols, side="right")
    out = np.zeros((rows.size, cols.size), dtype=_block_dtype(layout))
    infos = [_pointinfo(chnkr) for chnkr in layout["chunkers"]]
    weights = [chnkr.wts.reshape(-1, order="F") for chnkr in layout["chunkers"]]

    for itarg in range(len(layout["chunkers"])):
        row_pos = np.flatnonzero(row_chunk == itarg)
        if row_pos.size == 0:
            continue
        local_rows = rows[row_pos] - layout["row_offsets"][itarg]
        rowdim = int(layout["rowdims"][itarg])
        targ_dofs = (local_rows // rowdim) * rowdim + (local_rows % rowdim)
        for isrc in range(len(layout["chunkers"])):
            col_pos = np.flatnonzero(col_chunk == isrc)
            if col_pos.size == 0:
                continue
            local_cols = cols[col_pos] - layout["col_offsets"][isrc]
            coldim = int(layout["coldims"][isrc])
            src_dofs = (local_cols // coldim) * coldim + (local_cols % coldim)
            block = _subblock_from_dofs(
                layout["kernels"][itarg, isrc],
                infos[isrc],
                weights[isrc],
                src_dofs,
                coldim,
                infos[itarg],
                weights[itarg],
                targ_dofs,
                rowdim,
                l2scale=l2scale,
            )
            out[np.ix_(row_pos, col_pos)] = block
    return _overwrite_sparse(out, rows, cols, spmat)


def _block_layout(chnkobj: Any, kerns: Any, opdims: tuple[int, int] | ArrayLike | None) -> dict[str, Any]:
    chunkers = _chunker_sequence(chnkobj)
    kernels = np.asarray(kerns, dtype=object)
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
            opdims_mat = flat.reshape(2, nchunker, nchunker, order="F")
    else:
        opdims_mat = np.zeros((2, nchunker, nchunker), dtype=int)
        for itarg, targ in enumerate(chunkers):
            for isrc, src in enumerate(chunkers):
                op0, op1 = _opdims(src, kernels[itarg, isrc], None, targobj=targ)
                opdims_mat[:, itarg, isrc] = (op0, op1)

    for itarg in range(nchunker):
        for isrc in range(nchunker):
            op0, op1 = int(opdims_mat[0, itarg, isrc]), int(opdims_mat[1, itarg, isrc])
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
    for itarg, targ in enumerate(layout["chunkers"]):
        targ_info = _subset_info(_pointinfo(targ), np.array([0], dtype=np.int64))
        for isrc, src in enumerate(layout["chunkers"]):
            src_info = _subset_info(_pointinfo(src), np.array([0], dtype=np.int64))
            try:
                dtype = np.result_type(dtype, np.asarray(_eval_kernel(layout["kernels"][itarg, isrc], src_info, targ_info)).dtype)
            except Exception:
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
        items = list(np.ravel(obj, order="F"))
        if items and all(isinstance(item, Chunker) for item in items):
            return items
    edges = getattr(obj, "echnks", None)
    if edges is not None:
        items = list(edges)
        if items and all(isinstance(item, Chunker) for item in items):
            return items
    raise TypeError("expected a chunker sequence or chunkgraph-like object")


def _is_block_kernel_matrix(kern: Any) -> bool:
    if callable(kern):
        return False
    try:
        arr = np.asarray(kern, dtype=object)
    except Exception:
        return False
    return arr.ndim == 2 and arr.size > 0 and all(callable(item) for item in arr.flat)


def _pointinfo(obj: Any) -> Any:
    if isinstance(obj, Chunker):
        return _new_info(
            r=obj.r.reshape(obj.dim, obj.npt, order="F"),
            d=obj.d.reshape(obj.dim, obj.npt, order="F"),
            d2=obj.d2.reshape(obj.dim, obj.npt, order="F"),
            n=obj.n.reshape(obj.dim, obj.npt, order="F"),
            data=obj.data.reshape(obj.datadim, obj.npt, order="F") if obj.datadim else None,
        )
    merged = getattr(obj, "merged", None)
    if callable(merged):
        return _pointinfo(merged())
    if all(hasattr(obj, name) for name in ("r",)):
        r = np.asarray(obj.r)
        return _new_info(
            r=r.reshape(r.shape[0], -1),
            d=_optional_attr(obj, "d"),
            d2=_optional_attr(obj, "d2"),
            n=_optional_attr(obj, "n"),
            data=_optional_attr(obj, "data"),
        )
    if isinstance(obj, dict):
        r = np.asarray(obj["r"])
        return _new_info(
            r=r.reshape(r.shape[0], -1),
            d=_optional_item(obj, "d"),
            d2=_optional_item(obj, "d2"),
            n=_optional_item(obj, "n"),
            data=_optional_item(obj, "data"),
        )
    arr = np.asarray(obj, dtype=float)
    return _new_info(r=arr.reshape(arr.shape[0], -1))


def _proxy_info(pr: ArrayLike, ptau: ArrayLike, scale: float, ctr: np.ndarray) -> Any:
    pxy = np.asarray(pr, dtype=float).reshape(2, -1) * scale + ctr
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


def _optional_attr(obj: Any, name: str) -> np.ndarray | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return arr.reshape(arr.shape[0], -1)


def _optional_item(obj: dict[str, Any], name: str) -> np.ndarray | None:
    if name not in obj or obj[name] is None:
        return None
    arr = np.asarray(obj[name])
    if arr.size == 0:
        return None
    return arr.reshape(arr.shape[0], -1)


def _opdims(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | ArrayLike | None,
    *,
    targobj: Any | None = None,
) -> tuple[int, int]:
    if opdims is not None:
        arr = np.asarray(opdims, dtype=int).reshape(-1)
        if arr.size < 2:
            raise ValueError("opdims must contain two entries")
        return int(arr[0]), int(arr[1])
    kernel_dims = getattr(kern, "opdims", None)
    if kernel_dims is not None and tuple(kernel_dims) != (0, 0):
        return int(kernel_dims[0]), int(kernel_dims[1])
    src = _subset_info(_pointinfo(chnkr), np.array([0], dtype=np.int64))
    targ_info = _pointinfo(chnkr if targobj is None else targobj)
    targ = _subset_info(targ_info, np.array([0], dtype=np.int64))
    mat = np.asarray(_eval_kernel(kern, src, targ))
    return int(mat.shape[0]), int(mat.shape[1])


def _eval_kernel(kern: Callable[[Any, Any], np.ndarray], srcinfo: Any, targinfo: Any) -> np.ndarray:
    if hasattr(kern, "eval") and getattr(kern, "eval") is not None:
        return kern.eval(srcinfo, targinfo)
    return kern(srcinfo, targinfo)


def _overwrite_sparse(out: np.ndarray, rows: np.ndarray, cols: np.ndarray, spmat: spmatrix | None) -> np.ndarray:
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
    from ..operators import PointInfo

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
