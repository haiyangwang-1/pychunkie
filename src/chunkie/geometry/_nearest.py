"""Nearest-point helpers for chunker geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .. import lege
from .._legacy import warn_legacy_options

if TYPE_CHECKING:
    from ._chunker_class import Chunker


def chunk_nearparam(
    rval: ArrayLike,
    pts: ArrayLike,
    options: dict | None = None,
    t: ArrayLike | None = None,
    u: ArrayLike | None = None,
    *,
    max_iterations: int | None = None,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find nearest curve parameters on a single chunk."""

    option_values = warn_legacy_options(options, "chunk_nearparam")
    if max_iterations is not None:
        option_values["nitermax"] = max_iterations
    if threshold is not None:
        option_values["thresh"] = threshold
    maxnewt = int(option_values.get("nitermax", 15))
    thresh0 = float(option_values.get("thresh", 1.0e-14))

    r_arr = np.asarray(rval)
    if r_arr.ndim != 2:
        raise ValueError("rval must have shape (dim, k)")
    dim, k = r_arr.shape

    pts_arr = np.asarray(pts, dtype=r_arr.dtype).reshape(dim, -1)
    npts = pts_arr.shape[1]

    if t is None or u is None:
        t_arr, _, u_arr, _ = lege.exps(k)
    else:
        t_arr = np.asarray(t, dtype=float).reshape(k)
        u_arr = np.asarray(u)

    rc = u_arr @ r_arr.T
    drc = np.vstack((lege.derpol(rc), np.zeros((1, dim), dtype=rc.dtype)))
    d2rc = np.vstack((lege.derpol(drc), np.zeros((1, dim), dtype=rc.dtype)))
    cfs = np.concatenate((rc, drc, d2rc), axis=1)

    diffs = pts_arr[:, None, :] - r_arr[:, :, None]
    dist2_nodes = np.sum(np.abs(diffs) ** 2, axis=0)
    ipt = np.argmin(dist2_nodes, axis=0)
    dist2_best = dist2_nodes[ipt, np.arange(npts)]

    ts = np.zeros(npts, dtype=float)
    rs = np.zeros((dim, npts), dtype=r_arr.dtype)
    ds = np.zeros_like(rs)
    d2s = np.zeros_like(rs)
    dist2s = np.zeros(npts, dtype=float)

    thresh = thresh0 * (k * k * np.sum(np.abs(drc)) + k * np.sum(np.abs(rc)))
    thresh = max(float(thresh), np.finfo(float).eps)

    for idx in range(npts):
        ref = pts_arr[:, idx]
        t0 = float(t_arr[ipt[idx]])
        vals = np.asarray(lege.exev(np.array([t0]), cfs))[0]
        r0 = vals[:dim]
        d0 = vals[dim : 2 * dim]
        d20 = vals[2 * dim :]

        ts[idx] = t0
        rs[:, idx] = r0
        ds[:, idx] = d0
        d2s[:, idx] = d20
        dist2s[idx] = float(dist2_best[idx])

        rdiff = r0 - ref
        dprime = float(np.vdot(rdiff, d0).real)
        dprime2 = float((np.vdot(d0, d0) + np.vdot(rdiff, d20)).real)
        newton_success = False
        stable_iters = 0

        for _ in range(maxnewt):
            if abs(dprime2) <= np.finfo(float).eps:
                break
            dt = -dprime / dprime2
            t1 = min(max(t0 + dt, -1.0), 1.0)
            dt = t1 - t0
            t0 = t1

            vals = np.asarray(lege.exev(np.array([t0]), cfs))[0]
            r0 = vals[:dim]
            d0 = vals[dim : 2 * dim]
            d20 = vals[2 * dim :]
            rdiff = r0 - ref
            dprime = float(np.vdot(rdiff, d0).real)
            dprime2 = float((np.vdot(d0, d0) + np.vdot(rdiff, d20)).real)

            if min(abs(dprime), abs(dt)) < thresh:
                stable_iters += 1
            if stable_iters >= 3:
                newton_success = True
                break
            if (t0 == 1.0 and dprime < 0.0) or (t0 == -1.0 and dprime > 0.0):
                newton_success = True
                break

        dist2_newton = float(np.sum(np.abs(rdiff) ** 2))
        if dist2_newton <= dist2_best[idx]:
            ts[idx] = t0
            rs[:, idx] = r0
            ds[:, idx] = d0
            d2s[:, idx] = d20
            dist2s[idx] = dist2_newton
        else:
            newton_success = False

        if newton_success:
            continue

        t0 = float(t_arr[ipt[idx]])
        vals = np.asarray(lege.exev(np.array([t0]), cfs))[0]
        r0 = vals[:dim]
        d0 = vals[dim : 2 * dim]
        d20 = vals[2 * dim :]
        rdiff = r0 - ref
        dprime = float(np.vdot(rdiff, d0).real)
        dprime2 = max(float(np.vdot(d0, d0).real), np.finfo(float).eps)
        lam = dprime2
        dist0 = float(np.sum(np.abs(rdiff) ** 2))

        for _ in range(maxnewt):
            dt = -dprime / (dprime2 + lam)
            t1 = min(max(t0 + dt, -1.0), 1.0)
            vals1 = np.asarray(lege.exev(np.array([t1]), cfs))[0]
            r1 = vals1[:dim]
            d1 = vals1[dim : 2 * dim]
            d21 = vals1[2 * dim :]
            rdiff1 = r1 - ref
            dist1 = float(np.sum(np.abs(rdiff1) ** 2))

            if dist1 > dist0:
                lam *= 2.0
                continue

            t0 = t1
            r0 = r1
            d0 = d1
            d20 = d21
            rdiff = rdiff1
            dprime = float(np.vdot(rdiff, d0).real)
            dprime2 = max(float(np.vdot(d0, d0).real), np.finfo(float).eps)
            dist0 = dist1
            lam /= 3.0
            if abs(dprime) < thresh:
                break
            if (t0 == 1.0 and dprime < 0.0) or (t0 == -1.0 and dprime > 0.0):
                break

        ts[idx] = t0
        rs[:, idx] = r0
        ds[:, idx] = d0
        d2s[:, idx] = d20
        dist2s[idx] = dist0

    return ts, rs, ds, d2s, dist2s


def _bernstein_rectangle_info(chunker: Chunker, rho: float) -> np.ndarray:
    """Return MATLAB-style rectangle tests for Bernstein ellipse images."""

    ells = _bernstein_ellipse_images(chunker, rho)
    _, dc, _ = chunker.exps()
    p0 = _legendre_values(np.array([0.0]), chunker.k - 1).reshape(chunker.k)
    d0 = np.einsum("k,dkn->dn", p0, dc)
    d0_norm = np.sqrt(np.sum(d0**2, axis=0))
    d1s = d0 / d0_norm[None, :]
    d2s = np.vstack((d1s[1], -d1s[0]))

    d1c = np.einsum("dmn,dn->mn", ells, d1s)
    d2c = np.einsum("dmn,dn->mn", ells, d2s)

    rectinfo = np.zeros((2, 4, chunker.nch))
    rectinfo[:, 0, :] = d1s
    rectinfo[:, 1, :] = d2s
    rectinfo[0, 2, :] = np.min(d1c, axis=0)
    rectinfo[1, 2, :] = np.max(d1c, axis=0)
    rectinfo[0, 3, :] = np.min(d2c, axis=0)
    rectinfo[1, 3, :] = np.max(d2c, axis=0)
    return rectinfo


def _bernstein_ellipse_images(chunker: Chunker, rho: float) -> np.ndarray:
    nth = max(2 * chunker.nch, 20)
    theta = np.linspace(0.0, 2.0 * np.pi, nth + 1)[:-1]
    zrho = rho * np.exp(1j * theta)
    zell = (zrho + 1.0 / zrho) / 2.0
    zpols = _legendre_values(zell, chunker.k - 1).T
    rc, _, _ = chunker.exps()
    zcoef = rc[0] + 1j * rc[1]
    ell = zpols @ zcoef
    return np.stack((ell.real, ell.imag), axis=0)


def _legendre_values(xs: ArrayLike, degree: int) -> np.ndarray:
    xs_arr = np.asarray(xs)
    flat = xs_arr.reshape(-1)
    vals = np.zeros((degree + 1, flat.size), dtype=np.result_type(xs_arr, float))
    vals[0] = 1.0
    if degree >= 1:
        vals[1] = flat
    for k in range(1, degree):
        vals[k + 1] = ((2 * k + 1) * flat * vals[k] - k * vals[k - 1]) / (k + 1)
    return vals.reshape((degree + 1,) + xs_arr.shape)
