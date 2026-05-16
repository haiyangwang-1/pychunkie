"""Elasticity FMM selector wiring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from chunkie._layout import as_boundary_field_matrix, as_boundary_vector
from chunkie.geometry import PointInfo

from ._fmm_common import _laplace_log_moments
from ._fmm_laplace import _lap2d_fmm
from ._fmm_stokes import _stok2d_fmm
from .fmm import fmm2dpy


def _elast2d_fmm(
    kind: str, lam: float, mu: float
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if fmm2dpy is None:
        return None
    typ = kind.lower()
    if typ not in {
        "s",
        "single",
        "strac",
        "sgrad",
        "sg",
        "d",
        "double",
        "dalt",
        "daltgrad",
        "daltg",
        "dalttrac",
    }:
        return None
    beta = (lam + 3.0 * mu) / (4.0 * np.pi * mu * (lam + 2.0 * mu))
    gamma = -(lam + mu) / (4.0 * np.pi * mu * (lam + 2.0 * mu))
    eta = mu / (2.0 * np.pi * (lam + 2.0 * mu))
    zeta = (lam + mu) / (np.pi * (lam + 2.0 * mu))
    lap_d = _lap2d_fmm("d")
    lap_dgrad = _lap2d_fmm("dgrad")
    stok_d = _stok2d_fmm("d", mu)
    stok_dgrad = _stok2d_fmm("dgrad", mu)

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        src = PointInfo.from_any(source_info)
        targ = PointInfo.from_any(target_info)
        sig = as_boundary_field_matrix(
            sigma, 2, np.asarray(sigma).size // 2, name="elasticity density"
        )
        nt = targ.r.shape[1]
        if typ in {"s", "single", "sgrad", "sg", "strac"}:
            grad = typ in {"sgrad", "sg", "strac"}
            vals = _elast_single_fmm(eps, src, targ, sig, beta, gamma, want_grad=grad)
            if typ in {"s", "single"}:
                return vals
            grad_vals = as_boundary_field_matrix(vals, 4, nt, name="elasticity gradient")
            if typ in {"sgrad", "sg"}:
                return vals
            if targ.n is None:
                raise ValueError("target normals are required")
            return as_boundary_vector(
                _elastic_traction_from_grad(grad_vals, targ.n, lam, mu),
                name="elasticity traction",
            )

        if src.n is None:
            raise ValueError("source normals are required")
        if typ in {"d", "double"}:
            if lap_d is None or stok_d is None:
                raise ValueError(
                    "Laplace and Stokes FMM backends are required for elasticity double-layer FMM"
                )
            vel = (
                -zeta
                * np.pi
                * as_boundary_field_matrix(
                    stok_d(eps, src, targ, as_boundary_vector(sig)),
                    2,
                    nt,
                    name="Stokes velocity",
                )
            )
            norm = as_boundary_vector(lap_d(eps, src, targ, sig[0]), name="normal component")
            rot_x = as_boundary_vector(
                lap_d(eps, _with_normals(src, np.vstack((-src.n[1], src.n[0]))), targ, sig[1]),
                name="rotated x component",
            )
            rot_y = as_boundary_vector(
                lap_d(eps, _with_normals(src, np.vstack((src.n[1], -src.n[0]))), targ, sig[0]),
                name="rotated y component",
            )
            norm_y = as_boundary_vector(lap_d(eps, src, targ, sig[1]), name="normal y component")
            vel[0] += -eta * (2.0 * np.pi) * (norm + rot_x)
            vel[1] += -eta * (2.0 * np.pi) * (rot_y + norm_y)
            return as_boundary_vector(vel, name="elasticity velocity")

        if typ in {"dalt", "daltgrad", "daltg", "dalttrac"}:
            if (
                lap_d is None
                or stok_d is None
                or (typ not in {"dalt"} and (lap_dgrad is None or stok_dgrad is None))
            ):
                raise ValueError(
                    "Laplace and Stokes FMM backends are required for elasticity alternate double-layer FMM"
                )
            if typ == "dalt":
                vel = (
                    -zeta
                    * np.pi
                    * as_boundary_field_matrix(
                        stok_d(eps, src, targ, as_boundary_vector(sig)),
                        2,
                        nt,
                        name="Stokes velocity",
                    )
                )
                vel[0] += -4.0 * np.pi * eta * as_boundary_vector(lap_d(eps, src, targ, sig[0]))
                vel[1] += -4.0 * np.pi * eta * as_boundary_vector(lap_d(eps, src, targ, sig[1]))
                return as_boundary_vector(vel, name="elasticity velocity")

            grad_vals = (
                -zeta
                * np.pi
                * as_boundary_field_matrix(
                    stok_dgrad(eps, src, targ, as_boundary_vector(sig)),
                    4,
                    nt,
                    name="Stokes gradient",
                )
            )
            gx = as_boundary_field_matrix(lap_dgrad(eps, src, targ, sig[0]), 2, nt)
            gy = as_boundary_field_matrix(lap_dgrad(eps, src, targ, sig[1]), 2, nt)
            grad_vals[0] += -4.0 * np.pi * eta * gx[0]
            grad_vals[1] += -4.0 * np.pi * eta * gx[1]
            grad_vals[2] += -4.0 * np.pi * eta * gy[0]
            grad_vals[3] += -4.0 * np.pi * eta * gy[1]
            if typ in {"daltgrad", "daltg"}:
                return as_boundary_vector(grad_vals, name="elasticity gradient")
            if targ.n is None:
                raise ValueError("target normals are required")
            return as_boundary_vector(
                _elastic_traction_from_grad(grad_vals, targ.n, lam, mu),
                name="elasticity traction",
            )

        raise ValueError(f"Unknown elasticity FMM selector {kind!r}")

    return fmm_eval


def _elast_single_fmm(
    eps: float,
    source: Any,
    target: Any,
    sig: np.ndarray,
    beta: float,
    gamma: float,
    *,
    want_grad: bool,
) -> np.ndarray:
    sx = source.r[0]
    sy = source.r[1]
    tx = target.r[0]
    ty = target.r[1]
    a = sig[0]
    b = sig[1]
    charges = np.vstack((a, b, a * sx, a * sy, b * sx, b * sy))
    pot, grad, hess = _laplace_log_moments(eps, source, target, charges, pgt=3 if want_grad else 2)
    pa, pb = pot[0], pot[1]
    ga, gb, gasx, gasy, gbsx, gbsy = grad

    axx_a = tx * ga[0] - gasx[0]
    axy_a = ty * ga[0] - gasy[0]
    axy_b = ty * gb[0] - gbsy[0]
    ayy_b = ty * gb[1] - gbsy[1]

    ux = beta * pa + 0.5 * gamma * np.sum(a) + gamma * (axx_a + axy_b)
    uy = beta * pb + 0.5 * gamma * np.sum(b) + gamma * (axy_a + ayy_b)

    if not want_grad:
        return as_boundary_vector(np.vstack((ux, uy)), name="elasticity single-layer velocity")

    ha, hb, hasx, hasy, hbsx, hbsy = hess
    dux_dx = beta * ga[0] + gamma * ((ga[0] + tx * ha[0] - hasx[0]) + (ty * hb[0] - hbsy[0]))
    dux_dy = beta * ga[1] + gamma * ((tx * ha[1] - hasx[1]) + (gb[0] + ty * hb[1] - hbsy[1]))
    duy_dx = beta * gb[0] + gamma * ((ty * ha[0] - hasy[0]) + (ty * hb[1] - hbsy[1]))
    duy_dy = beta * gb[1] + gamma * (
        (ga[0] + ty * ha[1] - hasy[1]) + (gb[1] + ty * hb[2] - hbsy[2])
    )
    return as_boundary_vector(
        np.vstack((dux_dx, dux_dy, duy_dx, duy_dy)),
        name="elasticity single-layer gradient",
    )


def _elastic_traction_from_grad(
    grad: np.ndarray, normals: np.ndarray, lam: float, mu: float
) -> np.ndarray:
    nx = normals[0]
    ny = normals[1]
    du11, du12, du21, du22 = grad
    div = du11 + du22
    shear = float(mu) * (du12 + du21)
    tx = float(lam) * nx * div + shear * ny + 2.0 * float(mu) * du11 * nx
    ty = float(lam) * ny * div + shear * nx + 2.0 * float(mu) * du22 * ny
    return np.vstack((tx, ty))


def _with_normals(source: Any, normals: np.ndarray) -> Any:
    base = PointInfo.from_any(source)
    return PointInfo(r=base.r, d=base.d, d2=base.d2, n=np.asarray(normals), data=base.data)
