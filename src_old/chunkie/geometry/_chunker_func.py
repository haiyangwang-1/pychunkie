"""Parametric-curve chunker construction."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .. import lege
from ._chunker_class import Chunker
from ._chunker_options import _curve_outputs, _legacy_options, _pref_with_order, _set_option
from ._chunker_pref import ChunkerPref


def chunkerfunc(
    fcurve: Callable[[np.ndarray], Any],
    cparams: dict[str, Any] | None = None,
    pref: ChunkerPref | dict[str, Any] | None = None,
    *,
    order: int | None = None,
    closed: bool | None = None,
    interval: tuple[float, float] | None = None,
    tol: float | None = None,
    min_chunks: int | None = None,
    refine: bool | None = None,
    oversample: int | None = None,
    split_points: ArrayLike | None = None,
    max_chunk_length: float | None = None,
    level_restrict: str | None = None,
    level_restrict_factor: float | None = None,
    split_type: str | None = None,
) -> tuple[Chunker, np.ndarray]:
    """Create a chunker for a parameterized curve.

    The implementation follows MATLAB ``chunkerfunc``: initial parameter
    intervals are adaptively split until the curve and speed are spectrally
    resolved, then optional level restriction and oversampling are applied.

    ``fcurve(t)`` must return at least positions with shape ``(dim, len(t))``;
    first and second derivatives may also be returned. Common ``cparams`` are
    ``ta``/``tb`` for the parameter interval, ``ifclosed`` for topology,
    ``eps`` for resolution, ``nchmin`` for a minimum panel count, ``tsplits``
    for forced breakpoints, and ``maxchunklen`` for arclength control.
    """

    cparams = _legacy_options(cparams, "chunkerfunc cparams")
    if interval is not None:
        cparams["ta"], cparams["tb"] = interval
    _set_option(cparams, "ifclosed", closed)
    _set_option(cparams, "eps", tol)
    _set_option(cparams, "nchmin", min_chunks)
    _set_option(cparams, "ifrefine", refine)
    _set_option(cparams, "nover", oversample)
    _set_option(cparams, "tsplits", split_points)
    _set_option(cparams, "maxchunklen", max_chunk_length)
    _set_option(cparams, "lvlr", level_restrict)
    _set_option(cparams, "lvlrfac", level_restrict_factor)
    _set_option(cparams, "stype", split_type)
    p = _pref_with_order(pref, order)

    ta = float(cparams.get("ta", 0.0))
    tb = float(cparams.get("tb", 2.0 * np.pi))
    ifclosed = bool(cparams.get("ifclosed", True))
    nover = int(cparams.get("nover", 0))
    nchmin = int(cparams.get("nchmin", 0))
    tsplits = np.asarray(cparams.get("tsplits", []), dtype=float).reshape(-1)
    eps = float(cparams.get("eps", 1.0e-6))
    ifrefine = bool(cparams.get("ifrefine", True))
    lvlr = str(cparams.get("lvlr", "a")).lower()
    lvlrfac = float(cparams.get("lvlrfac", Chunker.lvlrfacdefault))
    maxchunklen = float(cparams.get("maxchunklen", np.inf))
    chsmall = np.asarray(cparams.get("chsmall", [np.inf, np.inf]), dtype=float).reshape(-1)
    if chsmall.size == 1:
        chsmall = np.repeat(chsmall, 2)
    if chsmall.size != 2:
        raise ValueError("chsmall must be scalar or length 2")

    if tb <= ta:
        raise ValueError("tb must be greater than ta")
    if np.any(tsplits < ta) or np.any(tsplits > tb):
        raise ValueError("tsplits outside interval of definition")

    first = _curve_outputs(fcurve, np.array([ta]))
    dim = first[0].shape[0]
    nout = min(len(first), 3)

    breaks = np.unique(np.concatenate(([ta], tsplits, [tb])))
    breaks.sort()
    if breaks.size < 2:
        raise ValueError("at least one parameter interval is required")

    if nchmin > 0:
        while breaks.size - 1 < nchmin:
            breaks = np.sort(np.concatenate((breaks, 0.5 * (breaks[:-1] + breaks[1:]))))

    if ifrefine:
        breaks = _adaptive_curve_breaks(
            fcurve, breaks, p.k, p.nchmax, dim, nout, eps, maxchunklen, chsmall, ifclosed
        )
    elif np.isfinite(maxchunklen):
        breaks = _maxlen_curve_breaks(fcurve, breaks, p.k, p.nchmax, dim, nout, maxchunklen)

    if lvlr in {"a", "t"}:
        breaks = _level_restrict_curve_breaks(
            fcurve, breaks, p.k, p.nchmax, dim, nout, ifclosed, lvlrfac, lvlr
        )
    elif lvlr not in {"n", "none"}:
        raise ValueError("lvlr must be 'a', 't', or 'n'")

    stype = str(cparams.get("stype", "a")).lower()
    for _ in range(max(nover, 0)):
        breaks = _oversample_curve_breaks(fcurve, breaks, p.k, p.nchmax, dim, nout, stype)

    ab = np.vstack((breaks[:-1], breaks[1:]))
    nch = ab.shape[1]
    if nch > p.nchmax:
        raise ValueError("CHUNKERFUNC: nchmax exceeded")

    p = ChunkerPref(p.nchmax, p.k, dim, max(p.nchstor, min(nch, p.nchmax)), p.verttol)
    chnkr = Chunker(p).addchunk(nch)
    dmat = lege.dermat(chnkr.k)

    for i in range(nch):
        a, b = ab[:, i]
        r, d, d2 = _curve_interval_outputs(fcurve, a, b, chnkr.tstor, dmat, dim, nout)

        chnkr.rstor[:, :, i] = r
        chnkr.dstor[:, :, i] = d
        chnkr.d2stor[:, :, i] = d2

    adjs = np.zeros((2, nch), dtype=int)
    adjs[0] = np.arange(0, nch)
    adjs[1] = np.arange(2, nch + 2)
    if ifclosed:
        adjs[0, 0] = nch
        adjs[1, -1] = 1
    else:
        adjs[0, 0] = -1
        adjs[1, -1] = -1
    chnkr.adj = adjs
    chnkr.recompute_geometry()
    _warn_if_closed_endpoint_mismatch(chnkr, eps, ifclosed, fcurve, ta, tb)
    return chnkr, ab


def _warn_if_closed_endpoint_mismatch(
    chunker: Chunker,
    eps: float,
    ifclosed: bool,
    fcurve: Callable[[np.ndarray], Any],
    ta: float,
    tb: float,
) -> None:
    if not ifclosed or chunker.nch == 0:
        return
    endpoint_outputs = _curve_outputs(fcurve, np.array([ta, tb]))
    left = endpoint_outputs[0][:, 0]
    right = endpoint_outputs[0][:, -1]
    bbox = chunker.max() - chunker.min()
    scale = float(max(np.max(np.abs(bbox)), 1.0))
    msgbase = "CHUNKERFUNC: "
    if np.linalg.norm(left - right) / scale > eps:
        warnings.warn(
            msgbase
            + "start and end points of curve parameterization are not the same to target precision "
            + "but ifclosed flag is true. Check curve parameterization or if not a closed curve "
            + "set flag appropriately and consider creating a chunkgraph object",
            UserWarning,
            stacklevel=2,
        )
        return
    if len(endpoint_outputs) >= 2:
        left_tangent = endpoint_outputs[1][:, 0]
        right_tangent = endpoint_outputs[1][:, -1]
        left_tangent = left_tangent / np.linalg.norm(left_tangent)
        right_tangent = right_tangent / np.linalg.norm(right_tangent)
    else:
        _, tend = chunker.chunkends([0, chunker.nch - 1])
        left_tangent = tend[:, 0, 0]
        right_tangent = tend[:, 1, -1]
    if np.linalg.norm(left_tangent - right_tangent) > eps * chunker.k:
        warnings.warn(
            msgbase
            + "unit tangent vectors at start and end points of curve are not the same to target precision "
            + "but ifclosed flag is true. Check curve parameterization or if not a closed curve "
            + "set flag appropriately and consider creating a chunkgraph object",
            UserWarning,
            stacklevel=2,
        )


def _curve_interval_outputs(
    fcurve: Callable[[np.ndarray], Any],
    a: float,
    b: float,
    nodes: np.ndarray,
    dmat: np.ndarray,
    dim: int,
    nout: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = (b - a) / 2.0
    ts = a + h * (nodes + 1.0)
    outs = _curve_outputs(fcurve, ts)
    r = outs[0]
    if r.shape != (dim, nodes.size):
        raise ValueError("curve position output has incompatible shape")
    if nout >= 2 and len(outs) >= 2:
        d = outs[1] * h
    else:
        d = r @ dmat.T
    if nout >= 3 and len(outs) >= 3:
        d2 = outs[2] * h * h
    else:
        d2 = d @ dmat.T
    return r, d, d2


def _adaptive_curve_breaks(
    fcurve: Callable[[np.ndarray], Any],
    breaks: np.ndarray,
    quadrature_order: int,
    nchmax: int,
    dim: int,
    nout: int,
    eps: float,
    maxchunklen: float,
    chsmall: np.ndarray,
    ifclosed: bool,
) -> np.ndarray:
    nodes, weights, u, _ = lege.exps(2 * quadrature_order)
    dmat = lege.dermat(2 * quadrature_order)
    for _ in range(max(nchmax, 1)):
        radius = _curve_radius_on_breaks(fcurve, breaks, nodes, dim)
        new_breaks = [float(breaks[0])]
        changed = False
        ninterval = breaks.size - 1
        for idx, (a, b) in enumerate(zip(breaks[:-1], breaks[1:], strict=False)):
            r, d, d2 = _curve_interval_outputs(fcurve, float(a), float(b), nodes, dmat, dim, nout)
            length = _local_curve_length(d, weights)
            unresolved = _curve_interval_unresolved(
                r, d, d2, weights, u, quadrature_order, eps, radius, nout, b - a
            )
            if np.isfinite(maxchunklen):
                unresolved = unresolved or length > maxchunklen
            if not ifclosed and idx == 0:
                unresolved = unresolved or length > chsmall[0]
            if not ifclosed and idx == ninterval - 1:
                unresolved = unresolved or length > chsmall[1]
            if unresolved:
                new_breaks.append(float(0.5 * (a + b)))
                changed = True
            new_breaks.append(float(b))
            if len(new_breaks) - 1 > nchmax:
                raise ValueError("CHUNKERFUNC: nchmax exceeded. Unable to resolve curve.")
        breaks = np.asarray(new_breaks, dtype=float)
        if not changed:
            return breaks
    raise RuntimeError("adaptive chunkerfunc refinement did not converge")


def _curve_interval_unresolved(
    r: np.ndarray,
    d: np.ndarray,
    d2: np.ndarray,
    weights: np.ndarray,
    u: np.ndarray,
    quadrature_order: int,
    eps: float,
    radius: float,
    nout: int,
    width: float,
) -> bool:
    speed = np.sqrt(np.sum(np.abs(d) ** 2, axis=0))
    speed_cfs = u @ speed
    low = float(np.sum(np.abs(speed_cfs[:quadrature_order]) ** 2))
    high = float(np.sum(np.abs(speed_cfs[quadrature_order:]) ** 2))
    speed_err = np.sqrt(high / max(low, np.finfo(float).eps) / quadrature_order)
    speed_bad = speed_err > eps if nout >= 2 else speed_err * width > eps * quadrature_order

    pos_cfs = u @ r.T
    pos_err = np.sqrt(
        np.max(np.sum(np.abs(pos_cfs[quadrature_order:, :]) ** 2, axis=0) / quadrature_order)
    )
    curve_bad = pos_err / max(radius, np.finfo(float).eps) > eps

    curvature_bad = False
    if r.shape[0] == 2:
        zd = d[0] + 1j * d[1]
        zdd = d2[0] + 1j * d2[1]
        denom = np.maximum(np.abs(zd) ** 2, np.finfo(float).eps)
        dkappa = np.imag(zdd * np.conj(zd)) / denom
        curvature_bad = float(np.dot(np.abs(dkappa), weights)) >= (2.0 * np.pi) / 3.0
    return bool(speed_bad or curve_bad or curvature_bad)


def _curve_radius_on_breaks(
    fcurve: Callable[[np.ndarray], Any], breaks: np.ndarray, nodes: np.ndarray, dim: int
) -> float:
    mins = np.full(dim, np.inf)
    maxs = np.full(dim, -np.inf)
    for a, b in zip(breaks[:-1], breaks[1:], strict=False):
        ts = float(a) + (float(b) - float(a)) * (nodes + 1.0) / 2.0
        r = _curve_outputs(fcurve, ts)[0]
        mins = np.minimum(mins, np.min(r, axis=1))
        maxs = np.maximum(maxs, np.max(r, axis=1))
    radius = float(np.max(maxs - mins))
    return radius if radius > 0.0 else 1.0


def _maxlen_curve_breaks(
    fcurve: Callable[[np.ndarray], Any],
    breaks: np.ndarray,
    quadrature_order: int,
    nchmax: int,
    dim: int,
    nout: int,
    maxchunklen: float,
) -> np.ndarray:
    nodes, weights, _, _ = lege.exps(quadrature_order)
    dmat = lege.dermat(quadrature_order)
    for _ in range(max(nchmax, 1)):
        new_breaks = [float(breaks[0])]
        changed = False
        for a, b in zip(breaks[:-1], breaks[1:], strict=False):
            _, d, _ = _curve_interval_outputs(fcurve, float(a), float(b), nodes, dmat, dim, nout)
            if _local_curve_length(d, weights) > maxchunklen:
                new_breaks.append(float(0.5 * (a + b)))
                changed = True
            new_breaks.append(float(b))
            if len(new_breaks) - 1 > nchmax:
                raise ValueError("CHUNKERFUNC: nchmax exceeded while enforcing maxchunklen")
        breaks = np.asarray(new_breaks, dtype=float)
        if not changed:
            return breaks
    raise RuntimeError("maxchunklen chunkerfunc refinement did not converge")


def _level_restrict_curve_breaks(
    fcurve: Callable[[np.ndarray], Any],
    breaks: np.ndarray,
    quadrature_order: int,
    nchmax: int,
    dim: int,
    nout: int,
    ifclosed: bool,
    lvlrfac: float,
    lvlr: str,
) -> np.ndarray:
    nodes, weights, _, _ = lege.exps(quadrature_order)
    dmat = lege.dermat(quadrature_order)
    for _ in range(1000):
        lengths = (
            np.diff(breaks)
            if lvlr == "t"
            else np.array(
                [
                    _curve_interval_length(
                        fcurve, float(a), float(b), nodes, weights, dmat, dim, nout
                    )
                    for a, b in zip(breaks[:-1], breaks[1:], strict=False)
                ]
            )
        )
        flags = np.zeros(lengths.size, dtype=bool)
        for idx, length in enumerate(lengths):
            left = lengths[idx - 1] if idx > 0 else (lengths[-1] if ifclosed else length)
            right = (
                lengths[idx + 1] if idx + 1 < lengths.size else (lengths[0] if ifclosed else length)
            )
            flags[idx] = length > lvlrfac * left or length > lvlrfac * right
        if not np.any(flags):
            return breaks
        new_breaks = [float(breaks[0])]
        for idx, (a, b) in enumerate(zip(breaks[:-1], breaks[1:], strict=False)):
            if flags[idx]:
                new_breaks.append(float(0.5 * (a + b)))
            new_breaks.append(float(b))
            if len(new_breaks) - 1 > nchmax:
                raise ValueError("CHUNKERFUNC: nchmax exceeded during level restriction")
        breaks = np.asarray(new_breaks, dtype=float)
    raise RuntimeError("level-restriction chunkerfunc refinement did not converge")


def _oversample_curve_breaks(
    fcurve: Callable[[np.ndarray], Any],
    breaks: np.ndarray,
    quadrature_order: int,
    nchmax: int,
    dim: int,
    nout: int,
    stype: str,
) -> np.ndarray:
    nodes, weights, _, _ = lege.exps(quadrature_order)
    dmat = lege.dermat(quadrature_order)
    new_breaks = [float(breaks[0])]
    for a, b in zip(breaks[:-1], breaks[1:], strict=False):
        if stype.startswith("a"):
            mid = _curve_arclength_midpoint(
                fcurve, float(a), float(b), nodes, weights, dmat, dim, nout
            )
        else:
            mid = float(0.5 * (a + b))
        new_breaks.extend((mid, float(b)))
        if len(new_breaks) - 1 > nchmax:
            raise ValueError("CHUNKERFUNC: nchmax exceeded while oversampling")
    return np.asarray(new_breaks, dtype=float)


def _curve_arclength_midpoint(
    fcurve: Callable[[np.ndarray], Any],
    a: float,
    b: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    dmat: np.ndarray,
    dim: int,
    nout: int,
) -> float:
    total = _curve_interval_length(fcurve, a, b, nodes, weights, dmat, dim, nout)
    target = 0.5 * total
    lo = a
    hi = b
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        left = _curve_interval_length(fcurve, a, mid, nodes, weights, dmat, dim, nout)
        if abs(left - target) <= 1e-13 * max(total, 1.0):
            return mid
        if left < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _curve_interval_length(
    fcurve: Callable[[np.ndarray], Any],
    a: float,
    b: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    dmat: np.ndarray,
    dim: int,
    nout: int,
) -> float:
    _, d, _ = _curve_interval_outputs(fcurve, a, b, nodes, dmat, dim, nout)
    return _local_curve_length(d, weights)


def _local_curve_length(d: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(np.sqrt(np.sum(np.abs(d) ** 2, axis=0)), weights))
