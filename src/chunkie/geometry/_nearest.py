"""Nearest-point helpers for chunker geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .. import lege


def chunk_nearparam(
    rval: ArrayLike,
    pts: ArrayLike,
    options: dict | None = None,
    t: ArrayLike | None = None,
    u: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find nearest curve parameters on a single chunk."""

    option_values = {} if options is None else dict(options)
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
