"""Helmholtz FMM selector wiring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from chunkie._layout import as_boundary_vector
from chunkie.geometry import PointInfo

from ._fmm_common import _sum_raw_fmm
from .fmm import fmm2dpy


def _helm2d_fmm(
    kind: str, zk: complex, coefs: Any | None = None
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if fmm2dpy is None:
        return None
    typ = kind.lower()
    if typ in {"c", "combined"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_helm2d_fmm("d", zk), _helm2d_fmm("s", zk), c[0], c[1])
    if typ in {"cp", "cprime"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_helm2d_fmm("dp", zk), _helm2d_fmm("sp", zk), c[0], c[1])
    if typ not in {
        "s",
        "single",
        "d",
        "double",
        "sp",
        "sprime",
        "stau",
        "st",
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
        if typ in {"s", "single", "sgrad", "sg", "sp", "sprime", "stau", "st"}:
            out = fmm2dpy.hfmm2d(eps=eps, zk=zk, sources=src.r, charges=sig, targets=targ.r, pgt=2)
        else:
            if src.n is None:
                raise ValueError("source normals are required")
            pgt = 2 if typ in {"dgrad", "dg", "dp", "dprime"} else 1
            out = fmm2dpy.hfmm2d(
                eps=eps, zk=zk, sources=src.r, dipstr=sig, dipvec=src.n, targets=targ.r, pgt=pgt
            )
        if typ in {"s", "single", "d", "double"}:
            return as_boundary_vector(out.pottarg, name="Helmholtz FMM potential")
        grad = np.asarray(out.gradtarg)
        if typ in {"sgrad", "sg", "dgrad", "dg"}:
            return as_boundary_vector(grad, name="Helmholtz FMM gradient")
        if typ in {"sp", "sprime", "dp", "dprime"}:
            if targ.n is None:
                raise ValueError("target normals are required")
            return grad[0] * targ.n[0] + grad[1] * targ.n[1]
        if typ in {"stau", "st"}:
            if targ.d is None:
                raise ValueError("target tangents are required")
            speed = np.sqrt(targ.d[0] ** 2 + targ.d[1] ** 2)
            return (grad[0] * targ.d[0] + grad[1] * targ.d[1]) / speed
        raise ValueError(f"Unknown Helmholtz FMM selector {kind!r}")

    return fmm_eval
