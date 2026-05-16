"""Laplace FMM selector wiring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from chunkie._layout import as_boundary_vector
from chunkie.geometry import PointInfo

from ._fmm_common import _sum_raw_fmm
from .fmm import fmm2dpy


def _lap2d_fmm(
    kind: str, coefs: Any | None = None
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if fmm2dpy is None:
        return None
    typ = kind.lower()
    if typ in {"c", "combined"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_lap2d_fmm("d"), _lap2d_fmm("s"), c[0], c[1])
    if typ in {"cp", "cprime"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_lap2d_fmm("dp"), _lap2d_fmm("sp"), c[0], c[1])
    if typ in {"cg", "cgrad"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_lap2d_fmm("dg"), _lap2d_fmm("sg"), c[0], c[1])
    if typ not in {
        "s",
        "single",
        "d",
        "double",
        "sp",
        "sprime",
        "st",
        "stau",
        "hilb",
        "sgrad",
        "sg",
        "dgrad",
        "dg",
        "dp",
        "dprime",
    }:
        return None

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        src = PointInfo.from_any(source_info)
        targ = PointInfo.from_any(target_info)
        sig = as_boundary_vector(sigma, name="density")
        if typ in {"s", "single", "sgrad", "sg", "sp", "sprime", "st", "stau"}:
            out = fmm2dpy.lfmm2d(eps=eps, sources=src.r, charges=sig, targets=targ.r, pgt=2)
        elif typ in {"hilb"}:
            if src.n is None:
                raise ValueError("source normals are required")
            dipvec = np.vstack((-src.n[1], src.n[0]))
            out = fmm2dpy.lfmm2d(
                eps=eps, sources=src.r, dipstr=2.0 * sig, dipvec=dipvec, targets=targ.r, pgt=1
            )
        else:
            if src.n is None:
                raise ValueError("source normals are required")
            out = fmm2dpy.lfmm2d(
                eps=eps, sources=src.r, dipstr=sig, dipvec=src.n, targets=targ.r, pgt=3
            )
        scale = -1.0 / (2.0 * np.pi)
        if typ in {"s", "single", "d", "double", "hilb"}:
            return np.real_if_close(
                scale * as_boundary_vector(out.pottarg, name="Laplace FMM potential")
            )
        grad = scale * np.asarray(out.gradtarg)
        if typ in {"sgrad", "sg", "dgrad", "dg"}:
            return np.real_if_close(as_boundary_vector(grad, name="Laplace FMM gradient"))
        if typ in {"sp", "sprime"}:
            if targ.n is None:
                raise ValueError("target normals are required")
            return np.real_if_close(grad[0] * targ.n[0] + grad[1] * targ.n[1])
        if typ in {"st", "stau"}:
            if targ.n is None:
                raise ValueError("target normals are required")
            return np.real_if_close(-grad[0] * targ.n[1] + grad[1] * targ.n[0])
        if typ in {"dp", "dprime"}:
            if targ.n is None:
                raise ValueError("target normals are required")
            return np.real_if_close(grad[0] * targ.n[0] + grad[1] * targ.n[1])
        raise ValueError(f"Unknown Laplace FMM selector {kind!r}")

    return fmm_eval
