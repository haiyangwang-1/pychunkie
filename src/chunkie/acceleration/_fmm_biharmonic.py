"""Biharmonic FMM selector wiring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from chunkie._layout import as_boundary_field_matrix, as_boundary_vector
from chunkie.geometry import PointInfo

from ._fmm_common import _laplace_log_moments
from .fmm import fmm2dpy


def _biharm2d_fmm(kind: str) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    typ = kind.lower()
    if fmm2dpy is None:
        return None
    if typ not in {
        "s",
        "single",
        "d",
        "double",
        "sp",
        "sprime",
        "sgrad",
        "sg",
        "shess",
        "hess",
        "lap",
        "slap",
        "laplacian",
    }:
        return None

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        src = PointInfo.from_any(source_info)
        targ = PointInfo.from_any(target_info)
        sig = as_boundary_vector(sigma, name="density")
        sx = src.r[0]
        sy = src.r[1]
        tx = targ.r[0]
        ty = targ.r[1]
        nt = targ.r.shape[1]

        if typ in {"d", "double"}:
            if src.n is None:
                raise ValueError("source normals are required")
            nx = src.n[0]
            ny = src.n[1]
            charges = np.vstack((sig * nx, sig * ny, sig * (nx * sx + ny * sy)))
            pot, _, _ = _laplace_log_moments(eps, src, targ, charges, pgt=1)
            moment = tx * pot[0] + ty * pot[1] - pot[2]
            affine = (
                tx * np.sum(sig * nx) + ty * np.sum(sig * ny) - np.sum(sig * (nx * sx + ny * sy))
            )
            return -(2.0 * moment + affine) / (8.0 * np.pi)

        if typ in {"sp", "sprime"}:
            if targ.n is None:
                raise ValueError("target normals are required")
            grad = as_boundary_field_matrix(
                _biharm2d_fmm("sgrad")(eps, src, targ, sig),
                2,
                nt,
                name="biharmonic gradient",
            )
            return grad[0] * targ.n[0] + grad[1] * targ.n[1]

        if typ in {"lap", "slap", "laplacian"}:
            pot, _, _ = _laplace_log_moments(eps, src, targ, sig, pgt=2)
            return (pot[0] + np.sum(sig)) / (2.0 * np.pi)

        charges = np.vstack((sig, sig * sx, sig * sy, sig * (sx * sx + sy * sy)))
        pot, grad, _ = _laplace_log_moments(eps, src, targ, charges, pgt=2)
        p0, px, py, p2 = pot
        g0, gx, gy, g2 = grad
        t2 = tx * tx + ty * ty

        if typ in {"s", "single"}:
            vals = t2 * p0 - 2.0 * tx * px - 2.0 * ty * py + p2
            return vals / (8.0 * np.pi)

        if typ in {"sgrad", "sg"}:
            out_x = (
                2.0 * tx * p0 + t2 * g0[0] - 2.0 * px - 2.0 * tx * gx[0] - 2.0 * ty * gy[0] + g2[0]
            )
            out_y = (
                2.0 * ty * p0 + t2 * g0[1] - 2.0 * tx * gx[1] - 2.0 * py - 2.0 * ty * gy[1] + g2[1]
            )
            return as_boundary_vector(
                np.vstack((out_x, out_y)) / (8.0 * np.pi),
                name="biharmonic gradient",
            )

        rx2 = tx * g0[0] - gx[0]
        rxy = ty * g0[0] - gy[0]
        ry2 = ty * g0[1] - gy[1]
        total = np.sum(sig)
        hxx = 2.0 * p0 + total + 2.0 * rx2
        hxy = 2.0 * rxy
        hyy = 2.0 * p0 + total + 2.0 * ry2
        if typ in {"shess", "hess"}:
            return as_boundary_vector(
                np.vstack((hxx, hxy, hyy)) / (8.0 * np.pi),
                name="biharmonic hessian",
            )
        return (hxx + hyy) / (8.0 * np.pi)

    return fmm_eval
