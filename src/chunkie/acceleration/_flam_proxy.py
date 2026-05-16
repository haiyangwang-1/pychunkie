"""Proxy geometry and proxy callback helpers for FLAM."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .. import lege
from .._layout import as_boundary_vector
from ._flam_common import (
    _as_chunker,
    _as_index_array,
    _eval_kernel,
    _new_info,
    _opdims,
    _perp_unit,
    _pointinfo,
    _proxy_info,
)
from ._flam_index import _subblock_from_dofs


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
