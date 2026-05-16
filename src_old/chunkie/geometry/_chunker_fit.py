"""Uniform and fitted chunker constructors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._chunker_class import Chunker
from ._chunker_func import chunkerfunc
from ._chunker_options import _LEGACY_OPTIONS_MARKER, _legacy_options, _set_option
from ._chunker_pref import ChunkerPref


def chunkerfuncuni(
    fcurve: Callable[[np.ndarray], Any],
    nch: int = 16,
    cparams: dict[str, Any] | None = None,
    pref: ChunkerPref | dict[str, Any] | None = None,
    *,
    order: int | None = None,
    closed: bool | None = None,
    interval: tuple[float, float] | None = None,
) -> Chunker:
    """Create a uniformly panelized chunker from a parametric curve."""

    def uniform_curve(t: np.ndarray) -> Any:
        raw = fcurve(t)
        if isinstance(raw, tuple) and len(raw) > 2:
            # MATLAB chunkerfuncuni computes local second derivatives spectrally.
            return raw[:2]
        return raw

    params = _legacy_options(cparams, "chunkerfuncuni cparams")
    if interval is not None:
        params["ta"], params["tb"] = interval
    _set_option(params, "ifclosed", closed)
    nch = int(nch)
    ta = float(params.get("ta", 0.0))
    tb = float(params.get("tb", 2.0 * np.pi))
    ifclosed = bool(params.get("ifclosed", True))
    params = {_LEGACY_OPTIONS_MARKER: True, "ta": ta, "tb": tb, "ifclosed": ifclosed}
    params["tsplits"] = np.linspace(ta, tb, nch + 1)[1:-1]
    params["ifrefine"] = False
    params["lvlr"] = "n"
    params["nover"] = 0
    chnkr, _ = chunkerfunc(uniform_curve, params, pref, order=order)
    return chnkr


def chunkerfit(
    points: ArrayLike,
    options: dict[str, Any] | None = None,
    *,
    closed: bool | None = None,
    split_at_points: bool | None = None,
    tol: float | None = None,
    order: int | None = None,
    pref: ChunkerPref | dict[str, Any] | None = None,
) -> Chunker:
    """Create a chunker by fitting a cubic spline through 2D points."""

    from scipy.interpolate import CubicSpline

    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[0] != 2:
        raise ValueError("Points must be specified as a 2xN matrix")
    geometry_options = _legacy_options(options, "chunkerfit options")
    _set_option(geometry_options, "ifclosed", closed)
    _set_option(geometry_options, "splitatpoints", split_at_points)
    if tol is not None:
        geometry_options.setdefault("cparams", {})["eps"] = tol
    if pref is not None:
        geometry_options["pref"] = pref
    if order is not None:
        p0 = ChunkerPref.from_any(geometry_options.get("pref", None))
        geometry_options["pref"] = ChunkerPref(
            p0.nchmax, int(order), p0.dim, p0.nchstor, p0.verttol
        )
    method = str(geometry_options.get("method", "spline")).lower()
    if method != "spline":
        raise ValueError(f"Unsupported method {method!r}")

    ifclosed = bool(geometry_options.get("ifclosed", True))
    pts = point_array
    if ifclosed and np.linalg.norm(point_array[:, 0] - point_array[:, -1]) > 1e-14:
        pts = np.column_stack((point_array, point_array[:, 0]))
    if pts.shape[1] < 3:
        raise ValueError("chunkerfit requires at least three points")

    seglen = np.sqrt(np.sum(np.diff(pts, axis=1) ** 2, axis=0))
    if np.any(seglen <= 0.0):
        raise ValueError("consecutive fit points must be distinct")
    t = np.concatenate(([0.0], np.cumsum(seglen)))

    bc_type = "periodic" if ifclosed else "not-a-knot"
    splx = CubicSpline(t, pts[0], bc_type=bc_type)
    sply = CubicSpline(t, pts[1], bc_type=bc_type)

    def splinefunc(tt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tt_arr = np.asarray(tt)
        return (
            np.vstack((splx(tt_arr), sply(tt_arr))),
            np.vstack((splx(tt_arr, 1), sply(tt_arr, 1))),
            np.vstack((splx(tt_arr, 2), sply(tt_arr, 2))),
        )

    cparams = dict(geometry_options.get("cparams", {}))
    cparams[_LEGACY_OPTIONS_MARKER] = True
    cparams["ifclosed"] = ifclosed
    cparams["ta"] = float(t[0])
    cparams["tb"] = float(t[-1])
    if bool(geometry_options.get("splitatpoints", False)):
        cparams["tsplits"] = t[1:-1]
    chnkr, _ = chunkerfunc(splinefunc, cparams, geometry_options.get("pref", None))
    # MATLAB chunkerfit's local ppdiff helper leaves fitted second derivatives
    # zero in the returned chunker; keep that observable behavior for parity.
    chnkr.d2 = np.zeros_like(chnkr.d2)
    return chnkr
