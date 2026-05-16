"""Stokes FMM selector wiring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from chunkie._layout import as_boundary_field_matrix, as_boundary_vector
from chunkie.geometry import PointInfo

from ._fmm_common import _sum_raw_fmm
from .fmm import fmm2dpy


def _stok2d_fmm(
    kind: str, mu: float = 1.0, coefs: Any | None = None
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if fmm2dpy is None:
        return None
    typ = kind.lower()
    if typ in {"c", "combined", "cvel", "cvelocity"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_stok2d_fmm("d", mu), _stok2d_fmm("s", mu), c[0], c[1])
    if typ in {"cpres", "cpressure"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_stok2d_fmm("dpres", mu), _stok2d_fmm("spres", mu), c[0], c[1])
    if typ in {"ctrac", "ctraction"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_stok2d_fmm("dtrac", mu), _stok2d_fmm("strac", mu), c[0], c[1])
    if typ in {"cg", "cgrad"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_stok2d_fmm("dgrad", mu), _stok2d_fmm("sgrad", mu), c[0], c[1])
    if typ in {"strac", "straction"}:
        return _stok2d_traction_fmm(_stok2d_fmm("spres", mu), _stok2d_fmm("sgrad", mu), mu)
    if typ in {"dtrac", "dtraction"}:
        return _stok2d_traction_fmm(_stok2d_fmm("dpres", mu), _stok2d_fmm("dgrad", mu), mu)
    if typ not in {
        "s",
        "single",
        "svel",
        "svelocity",
        "d",
        "double",
        "dvel",
        "dvelocity",
        "spres",
        "spressure",
        "dpres",
        "dpressure",
        "sgrad",
        "sg",
        "dgrad",
        "dg",
    }:
        return None

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        src = PointInfo.from_any(source_info)
        targ = PointInfo.from_any(target_info)
        sig = as_boundary_field_matrix(sigma, 2, np.asarray(sigma).size // 2, name="Stokes density")
        is_double = typ in {"d", "double", "dvel", "dvelocity", "dpres", "dpressure", "dgrad", "dg"}
        if is_double and src.n is None:
            raise ValueError("source normals are required")
        ifppregtarg = (
            3
            if typ in {"sgrad", "sg", "dgrad", "dg"}
            else 2
            if typ in {"spres", "spressure", "dpres", "dpressure"}
            else 1
        )
        kwargs = {"strslet": sig, "strsvec": src.n} if is_double else {"stoklet": sig}
        out = fmm2dpy.stfmm2d(
            eps=eps, sources=src.r, targets=targ.r, ifppregtarg=ifppregtarg, **kwargs
        )

        if typ in {"s", "single", "svel", "svelocity"}:
            scale = 1.0 / (2.0 * np.pi * float(mu))
            return scale * as_boundary_vector(np.asarray(out.pottarg)[0], name="Stokes velocity")
        if typ in {"d", "double", "dvel", "dvelocity"}:
            return (
                -1.0
                / (2.0 * np.pi)
                * as_boundary_vector(np.asarray(out.pottarg)[0], name="Stokes velocity")
            )
        if typ in {"spres", "spressure"}:
            return (
                1.0
                / (2.0 * np.pi)
                * as_boundary_vector(np.asarray(out.pretarg)[0], name="Stokes pressure")
            )
        if typ in {"dpres", "dpressure"}:
            return (
                -float(mu)
                / (2.0 * np.pi)
                * as_boundary_vector(np.asarray(out.pretarg)[0], name="Stokes pressure")
            )
        if typ in {"sgrad", "sg"}:
            scale = 1.0 / (2.0 * np.pi * float(mu))
            return scale * as_boundary_vector(np.asarray(out.gradtarg)[0], name="Stokes gradient")
        return (
            -1.0
            / (2.0 * np.pi)
            * as_boundary_vector(np.asarray(out.gradtarg)[0], name="Stokes gradient")
        )

    return fmm_eval


def _stok2d_traction_fmm(
    pressure_fmm: Callable[[float, Any, Any, np.ndarray], np.ndarray] | None,
    grad_fmm: Callable[[float, Any, Any, np.ndarray], np.ndarray] | None,
    mu: float,
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if pressure_fmm is None or grad_fmm is None:
        return None

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        targ = PointInfo.from_any(target_info)
        if targ.n is None:
            raise ValueError("target normals are required")
        pressure = as_boundary_vector(
            pressure_fmm(eps, source_info, targ, sigma), name="Stokes pressure"
        )
        grad = as_boundary_field_matrix(
            grad_fmm(eps, source_info, targ, sigma),
            4,
            targ.r.shape[1],
            name="Stokes gradient",
        )
        nx = targ.n[0]
        ny = targ.n[1]
        du11, du12, du21, du22 = grad
        mut = float(mu)
        tx = -pressure * nx + mut * (2.0 * du11 * nx + (du12 + du21) * ny)
        ty = -pressure * ny + mut * ((du21 + du12) * nx + 2.0 * du22 * ny)
        return as_boundary_vector(np.vstack((tx, ty)), name="Stokes traction")

    return fmm_eval
