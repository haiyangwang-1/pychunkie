"""Adaptive-special quadrature entry points.

The MATLAB implementation adaptively decides which close interactions need
replacement quadrature. This Python baseline exposes the same build path
and delegates the actual replacement blocks to ``quadggq``.

Use this path when target points are close to a source panel but are not just
the boundary self/neighbor cases covered by GGQ tables. The algorithm keeps the
smooth native matrix for far blocks, detects close target-panel pairs, and
replaces those blocks with recursively subdivided Gauss integrals. Product
quadrature can be used first for kernels with analytic split metadata; adaptive
Gauss remains the fallback when the side or split is unavailable.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie._layout import as_boundary_point_matrix

from .. import lege
from .._legacy import warn_legacy_options
from ..geometry import PointInfo
from ..geometry.chunker import Chunker
from . import ggq as quadggq
from . import native as quadnative
from . import panel as pquad

_ADAPTIVE_FAILURE_REASONS = {
    8: "maxdepth reached",
    16: "maxints exhausted",
}


def buildmat(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | None = None,
    options: dict[str, Any] | None = None,
) -> np.ndarray:
    """Assemble a dense matrix with adaptive close-panel replacements."""

    chunker = chunker
    kernel = kernel
    options = warn_legacy_options(options, "adaptive.buildmat")
    qtype = str(options.get("sing", getattr(kernel, "sing", "log") or "log")).lower()
    if qtype != "log":
        return quadggq.buildmat(
            chunker,
            kernel,
            opdims if opdims is not None else getattr(kernel, "opdims", None),
            singularity=qtype,
            ilist=options.get("ilist", None),
        )
    if opdims is None:
        opdims = getattr(kernel, "opdims", None)
    if opdims is None or opdims == (0, 0):
        raise ValueError("opdims must be provided for adaptive quadrature assembly")

    op0 = int(opdims[0])
    op1 = int(opdims[1])
    aux = quadggq.setup(chunker.k, qtype)
    mat = quadnative.buildmat(chunker, kernel, (op0, op1))
    nodes, weights = lege.exps(max(27, chunker.k + 1))[:2]
    bary = lege.barywts(chunker.k, chunker.tstor)
    ignored = (
        set()
        if options.get("ilist", None) is None
        else {int(idx) for idx in np.asarray(options.get("ilist"), dtype=int).reshape(-1)}
    )

    for src_chunk in range(chunker.nch):
        src_cols = _block_slice(src_chunk, chunker.k, op1)
        left, right = chunker.adj[:, src_chunk]
        for targ_chunk in (int(left) - 1, int(right) - 1):
            if targ_chunk < 0 or targ_chunk >= chunker.nch:
                continue
            if src_chunk in ignored and targ_chunk in ignored:
                continue
            rows = _block_slice(targ_chunk, chunker.k, op0)
            targinfo = _chunk_pointinfo(chunker, targ_chunk)
            mat[rows, src_cols] = _close_panel_matrix(
                chunker, src_chunk, targinfo, kernel, (op0, op1), nodes, weights, bary, options
            )

        if src_chunk not in ignored:
            rows = _block_slice(src_chunk, chunker.k, op0)
            mat[rows, src_cols] = quadggq.diagbuildmat(chunker, src_chunk, kernel, (op0, op1), aux)

    if bool(options.get("robust", False)):
        _apply_robust_close_corrections(
            mat, chunker, kernel, (op0, op1), nodes, weights, bary, options, ignored
        )

    return mat


def adapgausswts(
    chunker: Chunker,
    source_chunk: int,
    target: PointInfo,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    nodes: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    barywts: ArrayLike | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive source-panel quadrature weights for one chunk and target set.

    Each target gets an independent recursive integral over ``[-1, 1]`` on the
    source panel. A parent interval is accepted when splitting it in two changes
    the weighted kernel block by less than ``eps``; otherwise the children stay
    on the stack. The returned matrix maps original source-panel node values to
    target values, so callers can drop it directly into a global operator.
    """

    chunker = chunker
    src_chunk = source_chunk
    targinfo = target
    kernel = kernel
    options = warn_legacy_options(options, "adaptive.adapgausswts")
    eps = float(options.get("eps", 1.0e-12))
    maxints = int(options.get("maxints", 100000))
    maxdepth = int(options.get("maxdepth", 52))
    transinv = bool(options.get("transinv", True))
    recompute_source_normals = bool(options.get("recompute_source_normals", False))
    t = np.asarray(lege.exps(max(27, chunker.k + 1))[0] if nodes is None else nodes, dtype=float)
    w = np.asarray(
        lege.exps(max(27, chunker.k + 1))[1] if weights is None else weights, dtype=float
    )
    bw = np.asarray(
        lege.barywts(chunker.k, chunker.tstor) if barywts is None else barywts, dtype=float
    )

    op0 = int(opdims[0])
    op1 = int(opdims[1])
    ntarg = int(targinfo.r.shape[1])
    mat = np.zeros((op0 * ntarg, op1 * chunker.k), dtype=quadggq._kernel_dtype(chunker, kernel))
    maxrecs = np.zeros(ntarg, dtype=int)
    numints = np.zeros(ntarg, dtype=int)
    iers = np.zeros(ntarg, dtype=int)

    source = _chunk_source_arrays(chunker, src_chunk)
    for itarg in range(ntarg):
        one_targ = _single_target(targinfo, itarg)
        source_one = source
        if transinv:
            source_one = dict(source)
            source_one["r"] = source["r"] - one_targ.r
            one_targ = PointInfo(
                r=np.zeros_like(one_targ.r),
                d=one_targ.d,
                d2=one_targ.d2,
                n=one_targ.n,
                data=one_targ.data,
            )
        initial = _adaptive_panel_integral(
            -1.0,
            1.0,
            source_one,
            chunker.tstor,
            bw,
            one_targ,
            kernel,
            (op0, op1),
            t,
            w,
            recompute_source_normals,
        )
        stack = [(-1.0, 1.0, initial)]
        accum = np.zeros_like(initial)
        for count in range(1, maxints + 1):
            numints[itarg] = count
            maxrecs[itarg] = max(maxrecs[itarg], len(stack))
            a, b, parent = stack.pop()
            mid = 0.5 * (a + b)
            left = _adaptive_panel_integral(
                a,
                mid,
                source_one,
                chunker.tstor,
                bw,
                one_targ,
                kernel,
                (op0, op1),
                t,
                w,
                recompute_source_normals,
            )
            right = _adaptive_panel_integral(
                mid,
                b,
                source_one,
                chunker.tstor,
                bw,
                one_targ,
                kernel,
                (op0, op1),
                t,
                w,
                recompute_source_normals,
            )
            if np.max(np.abs(left + right - parent)) <= eps:
                accum = accum + left + right
                if not stack:
                    break
                continue
            if len(stack) + 2 > maxdepth:
                iers[itarg] = 8
                break
            stack.append((mid, b, right))
            stack.append((a, mid, left))
        else:
            iers[itarg] = 16
        mat[op0 * itarg : op0 * (itarg + 1), :] = accum
    return mat, maxrecs, numints, iers


def _adaptive_panel_integral(
    a: float,
    b: float,
    source: dict[str, np.ndarray | None],
    ct: np.ndarray,
    bw: np.ndarray,
    target: PointInfo,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    nodes: np.ndarray,
    weights: np.ndarray,
    recompute_source_normals: bool = False,
) -> np.ndarray:
    scale = (b - a) / 2.0
    shift = (b + a) / 2.0
    tt = scale * nodes + shift
    interp = _barycentric_matrix(ct, bw, tt)
    rint = source["r"] @ interp
    dint = source["d"] @ interp
    d2int = source["d2"] @ interp
    speed = np.sqrt(np.sum(np.abs(dint) ** 2, axis=0))
    src_n = (
        _normal_from_derivative(dint, speed) if recompute_source_normals else source["n"] @ interp
    )
    source = PointInfo(
        r=rint,
        d=dint,
        d2=d2int,
        n=src_n,
        data=None if source["data"] is None else source["data"] @ interp,
    )
    dsdt = scale * weights * speed
    kvals = quadggq._eval_kernel(kernel, source, target)
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    out = np.zeros((op0, op1 * ct.size), dtype=np.result_type(kvals, dsdt))
    for inode in range(tt.size):
        block = kvals[:, op1 * inode : op1 * (inode + 1)] * dsdt[inode]
        out += np.kron(interp[:, inode], block)
    return out


def _normal_from_derivative(d: np.ndarray, speed: np.ndarray) -> np.ndarray:
    if d.shape[0] != 2:
        raise ValueError(
            "source-normal recomputation is only implemented for two-dimensional chunkers"
        )
    n = np.empty_like(d)
    n[0] = d[1]
    n[1] = -d[0]
    return n / speed[None, :]


def _apply_robust_close_corrections(
    mat: np.ndarray,
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    nodes: np.ndarray,
    weights: np.ndarray,
    bary: np.ndarray,
    options: dict[str, Any],
    ignored: set[int],
) -> None:
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    points = as_boundary_point_matrix(chunker.r, chunker.dim, chunker.npt, name="positions")
    deriv = as_boundary_point_matrix(chunker.d, chunker.dim, chunker.npt, name="derivatives")
    deriv2 = as_boundary_point_matrix(
        chunker.d2, chunker.dim, chunker.npt, name="second derivatives"
    )
    normals = as_boundary_point_matrix(chunker.n, chunker.dim, chunker.npt, name="normals")
    data = (
        as_boundary_point_matrix(chunker.data, chunker.datadim, chunker.npt, name="data")
        if chunker.datadim
        else None
    )
    chunk_lengths = chunker.chunklen()
    for src_chunk in range(chunker.nch):
        if src_chunk in ignored:
            continue
        src_cols = _block_slice(src_chunk, chunker.k, op1)
        src_points = chunker.r[:, :, src_chunk]
        dist = np.sqrt(
            np.min(np.sum((points[:, :, None] - src_points[:, None, :]) ** 2, axis=0), axis=1)
        )
        close = np.flatnonzero(dist < chunk_lengths[src_chunk])
        if close.size == 0:
            continue
        left, right = chunker.adj[:, src_chunk]
        ignore_chunks = {src_chunk}
        if left > 0:
            ignore_chunks.add(int(left) - 1)
        if right > 0:
            ignore_chunks.add(int(right) - 1)
        ignore_points = set()
        for chunk in ignore_chunks:
            ignore_points.update(range(chunk * chunker.k, (chunk + 1) * chunker.k))
        fix = np.array([idx for idx in close if idx not in ignore_points], dtype=int)
        if fix.size == 0:
            continue
        targinfo = PointInfo(
            r=points[:, fix],
            d=deriv[:, fix],
            d2=deriv2[:, fix],
            n=normals[:, fix],
            data=None if data is None else data[:, fix],
        )
        submat = _close_panel_matrix(
            chunker, src_chunk, targinfo, kernel, opdims, nodes, weights, bary, options
        )
        for local, global_idx in enumerate(fix):
            rows = slice(op0 * global_idx, op0 * (global_idx + 1))
            mat[rows, src_cols] = submat[op0 * local : op0 * (local + 1), :]


def _barycentric_matrix(ct: np.ndarray, bw: np.ndarray, tt: np.ndarray) -> np.ndarray:
    diff = ct[:, None] - tt[None, :]
    out = np.empty((ct.size, tt.size), dtype=float)
    exact = np.isclose(diff, 0.0, atol=0.0, rtol=1e-14)
    for col in range(tt.size):
        hits = np.flatnonzero(exact[:, col])
        if hits.size:
            out[:, col] = 0.0
            out[hits[0], col] = 1.0
            continue
        vals = bw / diff[:, col]
        out[:, col] = vals / np.sum(vals)
    return out


def _chunk_source_arrays(chunker: Chunker, chunk: int) -> dict[str, np.ndarray | None]:
    return {
        "r": chunker.r[:, :, chunk],
        "d": chunker.d[:, :, chunk],
        "d2": chunker.d2[:, :, chunk],
        "n": chunker.n[:, :, chunk],
        "data": chunker.data[:, :, chunk] if chunker.datadim else None,
    }


def _chunk_pointinfo(chunker: Chunker, chunk: int) -> PointInfo:
    return PointInfo(
        r=chunker.r[:, :, chunk],
        d=chunker.d[:, :, chunk],
        d2=chunker.d2[:, :, chunk],
        n=chunker.n[:, :, chunk],
        data=chunker.data[:, :, chunk] if chunker.datadim else None,
    )


def _single_target(target_info: PointInfo, idx: int) -> PointInfo:
    return PointInfo(
        r=target_info.r[:, idx : idx + 1],
        d=None if target_info.d is None else target_info.d[:, idx : idx + 1],
        d2=None if target_info.d2 is None else target_info.d2[:, idx : idx + 1],
        n=None if target_info.n is None else target_info.n[:, idx : idx + 1],
        data=None if target_info.data is None else target_info.data[:, idx : idx + 1],
    )


def _close_panel_matrix(
    chunker: Chunker,
    source_chunk: int,
    target_info: PointInfo,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    nodes: np.ndarray,
    weights: np.ndarray,
    bary: np.ndarray,
    options: dict[str, Any],
) -> np.ndarray:
    pquad_mat, handled = _pquad_panel_matrix(
        chunker, source_chunk, target_info, kernel, opdims, options
    )
    if pquad_mat is not None and np.all(handled):
        return np.real_if_close(pquad_mat)

    adaptive, _, _, iers = adapgausswts(
        chunker,
        source_chunk,
        target_info,
        kernel,
        opdims,
        nodes,
        weights,
        bary,
        options,
    )
    warn_iers = iers
    if pquad_mat is not None and np.any(handled):
        op0 = int(opdims[0])
        rows = _target_rows(np.flatnonzero(handled), op0)
        adaptive = np.asarray(adaptive, dtype=np.result_type(adaptive.dtype, pquad_mat.dtype))
        adaptive[rows, :] = pquad_mat[rows, :]
        warn_iers = iers.copy()
        warn_iers[handled] = 0
    warn_adaptive_failures(
        warn_iers, src_chunk=source_chunk, context="adaptive close-panel quadrature", stacklevel=3
    )
    return adaptive


def _pquad_panel_matrix(
    chunker: Chunker,
    source_chunk: int,
    target_info: PointInfo,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    options: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray]:
    if not _pquad_enabled(options):
        return None, np.zeros(target_info.r.shape[1], dtype=bool)
    split_info = pquad.splitinfo_for_kernel(kernel=kernel)
    if split_info is None or tuple(split_info.opdims) != (int(opdims[0]), int(opdims[1])):
        return None, np.zeros(target_info.r.shape[1], dtype=bool)
    side_tol = options.get("side_tol", None)
    return pquad.panel_matrix_auto_side(
        chunker=chunker,
        source_chunk=source_chunk,
        target=target_info,
        split_info=split_info,
        side=_pquad_side(options),
        side_tol=None if side_tol is None else float(side_tol),
    )


def _pquad_enabled(options: dict[str, Any]) -> bool:
    if "forcepquad" in options:
        return _option_bool(options["forcepquad"])
    return _option_bool(options.get("usepquad", False))


def _pquad_side(options: dict[str, Any]) -> str | None:
    if "side" not in options or options["side"] is None:
        return None
    side = str(options["side"]).lower()
    if side not in {"i", "e"}:
        raise ValueError("side must be 'i' or 'e'")
    return side


def _option_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def warn_adaptive_failures(
    iers: ArrayLike,
    *,
    src_chunk: int | None = None,
    context: str = "adaptive quadrature",
    stacklevel: int = 2,
) -> None:
    """Warn when adaptive quadrature returned partial weights."""

    statuses = np.asarray(iers, dtype=int).reshape(-1)
    failed = statuses[statuses != 0]
    if failed.size == 0:
        return
    parts = []
    for code in np.unique(failed):
        reason = _ADAPTIVE_FAILURE_REASONS.get(int(code), f"status {int(code)}")
        parts.append(f"{reason}: {int(np.count_nonzero(failed == code))}")
    source = "" if src_chunk is None else f" on source chunk {int(src_chunk)}"
    warnings.warn(
        f"{context} did not converge for {failed.size} target(s){source} "
        f"({', '.join(parts)}); returning partially accumulated quadrature weights",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


def _target_rows(indices: np.ndarray, op0: int) -> np.ndarray:
    return (indices[:, None] * int(op0) + np.arange(int(op0))[None, :]).reshape(-1)


def _block_slice(chunk: int, quadrature_order: int, opdim: int) -> slice:
    start = chunk * quadrature_order * opdim
    return slice(start, start + quadrature_order * opdim)
