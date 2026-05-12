"""Helsing-Ojala product quadrature for close panel interactions.

This module ports the analytic panel-weight construction used by MATLAB
``chnk.pquadwts``. It is intentionally isolated from the public operator
dispatch for now: callers opt in by importing this module directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie import lege
from chunkie.chunker import Chunker
from chunkie.operators import PointInfo, pointinfo


SplitType = tuple[int, int, int, int]

SMOOTH: SplitType = (0, 0, 0, 0)
LOG: SplitType = (1, 0, 0, 0)
CAUCHY: SplitType = (0, 0, -1, 0)
HYPERSINGULAR: SplitType = (0, 0, -2, 0)
SUPERSINGULAR: SplitType = (0, 0, -3, 0)


@dataclass(frozen=True)
class SplitInfo:
    """Kernel split metadata for product quadrature assembly."""

    types: tuple[SplitType, ...]
    actions: tuple[str, ...]
    functions: Callable[[PointInfo, PointInfo], tuple[np.ndarray, ...]]
    opdims: tuple[int, int]


def pquadwts(
    chnkr: Chunker,
    src_chunk: int,
    targobj: PointInfo | dict[str, Any] | ArrayLike,
    types: Sequence[ArrayLike | SplitType],
    side: str,
    *,
    nodes: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    intp_ab: ArrayLike | None = None,
    intp: ArrayLike | None = None,
    ifup: bool = True,
) -> list[np.ndarray]:
    """Return product-quadrature weights for one source panel."""

    if src_chunk < 0 or src_chunk >= chnkr.nch:
        raise IndexError("source chunk index out of range")
    if chnkr.dim != 2:
        raise ValueError("product quadrature is implemented for 2D chunkers")

    k = chnkr.k
    if nodes is None or weights is None:
        t, w = lege.exps(2 * k)[:2]
    else:
        t = np.asarray(nodes, dtype=float).reshape(-1)
        w = np.asarray(weights, dtype=float).reshape(-1)
    if t.shape != w.shape:
        raise ValueError("nodes and weights must have the same shape")
    interp = lege.matrin(k, t)[0] if intp is None else np.asarray(intp)
    interp_ab = lege.matrin(k, np.array([-1.0, 1.0]))[0] if intp_ab is None else np.asarray(intp_ab)
    targ = pointinfo(targobj)
    return panel_pquadwts(
        chnkr.r,
        chnkr.d,
        chnkr.d2,
        chnkr.wts,
        src_chunk,
        targ.r,
        t,
        w,
        side,
        interp_ab,
        interp,
        types,
        ifup=ifup,
    )


def panel_matrix(
    chnkr: Chunker,
    src_chunk: int,
    targobj: PointInfo | dict[str, Any] | ArrayLike,
    splitinfo: SplitInfo,
    side: str,
    *,
    nodes: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> np.ndarray:
    """Assemble one source-panel product-quadrature matrix."""

    k = chnkr.k
    if nodes is None or weights is None:
        t, w = lege.exps(2 * k)[:2]
    else:
        t = np.asarray(nodes, dtype=float).reshape(-1)
        w = np.asarray(weights, dtype=float).reshape(-1)
    interp = lege.matrin(k, t)[0]
    interp_ab = lege.matrin(k, np.array([-1.0, 1.0]))[0]
    targ = pointinfo(targobj)
    weights_by_type = pquadwts(
        chnkr,
        src_chunk,
        targ,
        splitinfo.types,
        side,
        nodes=t,
        weights=w,
        intp_ab=interp_ab,
        intp=interp,
        ifup=True,
    )
    src = upsampled_sourceinfo(chnkr, src_chunk, interp)
    split_values = splitinfo.functions(src, targ)
    if len(split_values) != len(weights_by_type):
        raise ValueError("split function count does not match split types")

    op0, op1 = splitinfo.opdims
    out_up = np.zeros((op0 * targ.r.shape[1], op1 * t.size), dtype=_split_dtype(weights_by_type, split_values))
    for mat0, action, values in zip(weights_by_type, splitinfo.actions, split_values, strict=True):
        weighted = _apply_action(mat0, action)
        mat0opdim = np.kron(weighted, np.ones((op0, op1)))
        values_arr = np.asarray(values)
        if values_arr.shape != mat0opdim.shape:
            raise ValueError("split function returned an array with incompatible shape")
        out_up = out_up + mat0opdim * values_arr
    return out_up @ np.kron(interp, np.eye(op1))


def panel_pquadwts(
    r: np.ndarray,
    d: np.ndarray,
    d2: np.ndarray,
    wts: np.ndarray,
    src_chunk: int,
    rt: ArrayLike,
    t: ArrayLike,
    w: ArrayLike,
    side: str,
    intp_ab: ArrayLike,
    intp: ArrayLike,
    types: Sequence[ArrayLike | SplitType],
    *,
    ifup: bool = True,
) -> list[np.ndarray]:
    """Low-level product-quadrature weights for one panel."""

    side0 = str(side).lower()
    if side0 not in {"i", "e"}:
        raise ValueError("side must be 'i' or 'e'")

    r_arr = np.asarray(r)
    d_arr = np.asarray(d)
    d2_arr = np.asarray(d2)
    wts_arr = np.asarray(wts)
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    w_arr = np.asarray(w, dtype=float).reshape(-1)
    interp = np.asarray(intp)
    interp_ab = np.asarray(intp_ab)
    target = np.asarray(rt, dtype=float).reshape(2, -1)
    split_types = tuple(_split_type(type0) for type0 in types)
    nout = _required_special_count(split_types)

    z_nodes = r_arr[0, :, src_chunk] + 1j * r_arr[1, :, src_chunk]
    dz_nodes = d_arr[0, :, src_chunk] + 1j * d_arr[1, :, src_chunk]
    d2z_nodes = d2_arr[0, :, src_chunk] + 1j * d2_arr[1, :, src_chunk]

    xlohi = interp_ab @ z_nodes
    z_up = interp @ z_nodes
    dz_up = interp @ dz_nodes
    _ = interp @ d2z_nodes
    speed = np.abs(dz_up)
    normal = -1j * dz_up / speed
    wxp = w_arr * dz_up

    special: tuple[np.ndarray, ...] = ()
    if nout:
        special = sd_special_quad(target[0] + 1j * target[1], z_up, normal, wxp, xlohi[0], xlohi[1], side0, nout=nout)
        if not ifup:
            special = tuple(mat @ interp for mat in special)

    smooth_wts = w_arr * speed if ifup else wts_arr[:, src_chunk]
    out: list[np.ndarray] = []
    for type0 in split_types:
        if type0 == SMOOTH:
            out.append(np.ones((target.shape[1], smooth_wts.size), dtype=smooth_wts.dtype) * smooth_wts[None, :])
        elif type0 == LOG:
            out.append(special[0])
        elif type0 == CAUCHY:
            out.append(special[1])
        elif type0 == HYPERSINGULAR:
            out.append(special[2])
        elif type0 == SUPERSINGULAR:
            out.append(special[3])
        else:
            raise ValueError(f"unsupported split panel quadrature type {type0!r}")
    return out


def sd_special_quad(
    target: ArrayLike,
    source: ArrayLike,
    source_normal: ArrayLike,
    source_wxp: ArrayLike,
    a: complex,
    b: complex,
    side: str,
    *,
    nout: int = 4,
) -> tuple[np.ndarray, ...]:
    """Return Helsing-Ojala special weights ``As, A, A1, A2``."""

    if nout < 0 or nout > 4:
        raise ValueError("nout must be between 0 and 4")
    x_targ = np.asarray(target, dtype=complex).reshape(-1)
    y_src = np.asarray(source, dtype=complex).reshape(-1)
    normal = np.asarray(source_normal, dtype=complex).reshape(-1)
    wxp_src = np.asarray(source_wxp, dtype=complex).reshape(-1)
    if y_src.size != normal.size or y_src.size != wxp_src.size:
        raise ValueError("source, source_normal, and source_wxp sizes must agree")
    if y_src.size == 0 or x_targ.size == 0:
        return tuple(np.zeros((x_targ.size, y_src.size), dtype=complex) for _ in range(nout))

    zsc = (complex(b) - complex(a)) / 2.0
    zmid = (complex(b) + complex(a)) / 2.0
    y = (y_src - zmid) / zsc
    x = (x_targ - zmid) / zsc
    nsrc = y.size
    ntarg = x.size

    c = (1.0 - (-1.0) ** np.arange(1, nsrc + 1)) / np.arange(1, nsrc + 1)
    vander = np.ones((nsrc, nsrc), dtype=complex)
    for idx in range(1, nsrc):
        vander[:, idx] = vander[:, idx - 1] * y

    pvals = np.zeros((nsrc + 1, ntarg), dtype=complex)
    near = np.abs(x) <= 1.1
    far = ~near
    gamma = np.exp(1j * np.pi / 4.0)
    if str(side).lower() == "e":
        gamma = np.conj(gamma)
    elif str(side).lower() != "i":
        raise ValueError("side must be 'i' or 'e'")

    pvals[0] = np.log(gamma) + np.log((1.0 - x) / (gamma * (-1.0 - x)))
    if np.any(near):
        for idx in range(nsrc):
            pvals[idx + 1, near] = x[near] * pvals[idx, near] + c[idx]
    if np.any(far):
        xf = x[far]
        wxp = wxp_src / zsc
        pvals[nsrc, far] = np.sum((wxp * y**nsrc)[:, None] / (y[:, None] - xf[None, :]), axis=0)
        for matlab_idx in range(nsrc, 1, -1):
            pvals[matlab_idx - 1, far] = (pvals[matlab_idx, far] - c[matlab_idx - 1]) / xf

    qvals = np.zeros((nsrc, ntarg), dtype=complex)
    qvals[0::2] = pvals[1::2] - np.log((1.0 - x) * (-1.0 - x))[None, :]
    qvals[1::2] = pvals[2::2] - (np.log(gamma) + np.log((1.0 - x) / (gamma * (-1.0 - x))))[None, :]
    qvals *= (1.0 / np.arange(1, nsrc + 1))[:, None]

    solve_q = np.linalg.solve(vander.T, qvals).T
    as_weights = np.real(solve_q * np.conj(1j * normal)[None, :] * zsc) / (2.0 * np.pi * abs(zsc))
    as_weights = as_weights * abs(zsc) - np.log(abs(zsc)) / (2.0 * np.pi) * np.abs(wxp_src)[None, :]
    out: list[np.ndarray] = [as_weights]
    if nout == 1:
        return tuple(out)

    cauchy = np.linalg.solve(vander.T, pvals[:nsrc]).T * (1j / (2.0 * np.pi))
    out.append(cauchy)
    if nout == 2:
        return tuple(out)

    kidx = np.arange(nsrc, dtype=float)[:, None]
    signs = (-1.0) ** np.arange(nsrc, dtype=float)[:, None]
    rvals = -(1.0 / (1.0 - x)[None, :] + signs / (1.0 + x)[None, :])
    rvals += kidx * np.vstack((np.zeros((1, ntarg), dtype=complex), pvals[: nsrc - 1]))
    az = np.linalg.solve(vander.T, rvals).T * (1j / (2.0 * np.pi * zsc))
    out.append(az)
    if nout == 3:
        return tuple(out)

    svals = -(1.0 / (1.0 - x)[None, :] ** 2 - signs / (1.0 + x)[None, :] ** 2) / 2.0
    svals += kidx * np.vstack((np.zeros((1, ntarg), dtype=complex), rvals[: nsrc - 1])) / 2.0
    azz = np.linalg.solve(vander.T, svals).T * (1j / (2.0 * np.pi * zsc**2))
    out.append(azz)
    return tuple(out)


def upsampled_sourceinfo(chnkr: Chunker, src_chunk: int, intp: ArrayLike) -> PointInfo:
    """Return source point info interpolated to product-rule nodes."""

    interp = np.asarray(intp)
    r_up = (interp @ chnkr.r[:, :, src_chunk].T).T
    d_up = (interp @ chnkr.d[:, :, src_chunk].T).T
    d2_up = (interp @ chnkr.d2[:, :, src_chunk].T).T
    speed = np.sqrt(np.sum(np.abs(d_up) ** 2, axis=0))
    normal = np.vstack((d_up[1], -d_up[0])) / speed[None, :]
    data = (interp @ chnkr.data[:, :, src_chunk].T).T if chnkr.datadim else None
    return PointInfo(r=r_up, d=d_up, d2=d2_up, n=normal, data=data)


def splitinfo_for_kernel(kern: Any) -> SplitInfo | None:
    """Return split metadata for built-in scalar kernels when available."""

    name = str(getattr(kern, "name", "")).lower()
    kind = str(getattr(kern, "type", "")).lower()
    params = getattr(kern, "params", {}) or {}
    opdims = tuple(getattr(kern, "opdims", (1, 1)))
    if name == "laplace":
        return _laplace_splitinfo(kind, params.get("coefs", None))
    if name == "helmholtz":
        return _helmholtz_splitinfo(kind, params.get("zk", None), params.get("coefs", None), opdims)
    return None


def _laplace_splitinfo(kind: str, coefs: Any) -> SplitInfo | None:
    if kind in {"s", "single"}:
        return SplitInfo((LOG,), ("r",), lambda s, t: (_ones(t, s),), (1, 1))
    if kind in {"d", "double"}:
        return SplitInfo((CAUCHY,), ("r",), lambda s, t: (_ones(t, s),), (1, 1))
    if kind in {"c", "combined"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs).reshape(-1, order="F")

        def functions(s: PointInfo, t: PointInfo) -> tuple[np.ndarray, np.ndarray]:
            ones = _ones(t, s)
            return c[1] * ones, c[0] * ones

        return SplitInfo((LOG, CAUCHY), ("r", "r"), functions, (1, 1))
    return None


def _helmholtz_splitinfo(kind: str, zk: Any, coefs: Any, opdims: tuple[int, int]) -> SplitInfo | None:
    if zk is None or opdims != (1, 1):
        return None
    from . import helm2d

    if kind in {"s", "single"}:
        return SplitInfo((SMOOTH, LOG), ("r", "r"), lambda s, t: _helmholtz_s_split(helm2d, zk, s, t), (1, 1))
    if kind in {"d", "double"}:
        return SplitInfo((SMOOTH, LOG, CAUCHY), ("r", "r", "r"), lambda s, t: _helmholtz_d_split(helm2d, zk, s, t), (1, 1))
    if kind in {"c", "combined"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs).reshape(-1, order="F")
        return SplitInfo((SMOOTH, LOG, CAUCHY), ("r", "r", "r"), lambda s, t: _helmholtz_c_split(helm2d, zk, c, s, t), (1, 1))
    return None


def _helmholtz_s_split(helm2d: Any, zk: complex, src: PointInfo, targ: PointInfo) -> tuple[np.ndarray, np.ndarray]:
    seval = helm2d.kern(zk, src, targ, "s")
    dist = _complex_points(src)[None, :] - _complex_points(targ)[:, None]
    logeval = np.log(np.abs(dist))
    return seval + (2.0 / np.pi) * logeval * np.imag(seval), 4.0 * np.imag(seval)


def _helmholtz_d_split(helm2d: Any, zk: complex, src: PointInfo, targ: PointInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deval = helm2d.kern(zk, src, targ, "d")
    dist = _complex_points(src)[None, :] - _complex_points(targ)[:, None]
    logeval = np.log(np.abs(dist))
    cauchy = _complex_normals(src)[None, :] / dist
    return (
        deval + (2.0 / np.pi) * logeval * np.imag(deval) + np.real(cauchy) / (2.0 * np.pi),
        4.0 * np.imag(deval),
        _ones(targ, src),
    )


def _helmholtz_c_split(
    helm2d: Any,
    zk: complex,
    coefs: np.ndarray,
    src: PointInfo,
    targ: PointInfo,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s0, s1 = _helmholtz_s_split(helm2d, zk, src, targ)
    d0, d1, d2 = _helmholtz_d_split(helm2d, zk, src, targ)
    return coefs[0] * d0 + coefs[1] * s0, coefs[0] * d1 + coefs[1] * s1, coefs[0] * d2


def _split_type(value: ArrayLike | SplitType) -> SplitType:
    arr = tuple(int(v) for v in np.asarray(value, dtype=int).reshape(-1))
    if len(arr) != 4:
        raise ValueError("split panel quadrature types must have four integer entries")
    return arr  # type: ignore[return-value]


def _required_special_count(types: tuple[SplitType, ...]) -> int:
    nout = 0
    for type0 in types:
        if type0 == SMOOTH:
            continue
        if type0 == LOG:
            nout = max(nout, 1)
        elif type0 == CAUCHY:
            nout = max(nout, 2)
        elif type0 == HYPERSINGULAR:
            nout = max(nout, 3)
        elif type0 == SUPERSINGULAR:
            nout = max(nout, 4)
        else:
            joined = " ".join(str(v) for v in type0)
            raise ValueError(f"split panel quadrature type [{joined}] is not available")
    return nout


def _apply_action(mat: np.ndarray, action: str) -> np.ndarray:
    action0 = str(action).lower()
    if action0 == "r":
        return np.real(mat)
    if action0 == "i":
        return np.imag(mat)
    if action0 == "c":
        return mat
    raise ValueError("split action must be one of 'r', 'i', or 'c'")


def _split_dtype(weights: Sequence[np.ndarray], values: Sequence[np.ndarray]) -> np.dtype:
    dtype = np.dtype(float)
    for item in (*weights, *values):
        dtype = np.result_type(dtype, np.asarray(item).dtype)
    return dtype


def _complex_points(info: PointInfo) -> np.ndarray:
    return np.asarray(info.r[0]) + 1j * np.asarray(info.r[1])


def _complex_normals(info: PointInfo) -> np.ndarray:
    if info.n is None:
        raise ValueError("source normals are required")
    return np.asarray(info.n[0]) + 1j * np.asarray(info.n[1])


def _ones(targ: PointInfo, src: PointInfo) -> np.ndarray:
    return np.ones((targ.r.shape[1], src.r.shape[1]))
