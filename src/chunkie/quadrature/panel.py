"""Helsing-Ojala product quadrature for close panel interactions.

This module ports the analytic panel-weight construction used by MATLAB
``chnk.pquadwts``. Operator assembly uses these helpers for eligible close
target-panel replacements and falls back to adaptive or oversampled Gauss
quadrature when split metadata or side information is unavailable.

Panel product quadrature is different from GGQ tables: a kernel advertises a
local analytic split, such as smooth plus log or Cauchy pieces, and this module
builds weights for those singular pieces against one source panel. That is why
``SplitInfo`` stores both the split types and the callable that evaluates the
smooth coefficients multiplying each singular basis.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie import lege
from chunkie.geometry.chunker import Chunker
from chunkie.geometry import PointInfo


SplitType = tuple[int, int, int, int]

SMOOTH: SplitType = (0, 0, 0, 0)
LOG: SplitType = (1, 0, 0, 0)
CAUCHY: SplitType = (0, 0, -1, 0)
HYPERSINGULAR: SplitType = (0, 0, -2, 0)
SUPERSINGULAR: SplitType = (0, 0, -3, 0)


@dataclass(frozen=True)
class SplitInfo:
    """Kernel split metadata for product quadrature assembly.

    ``types`` describes the singular basis functions, ``actions`` says whether
    a weight matrix should be used directly or differentiated, and ``functions``
    evaluates the kernel-specific smooth coefficients on the upsampled panel.
    """

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
    """Return product-quadrature weights for one source panel.

    ``side`` is ``"i"`` or ``"e"`` for the interior or exterior branch cut.
    With ``ifup=True`` the returned matrices act on values on the supplied
    product-rule nodes.  With ``ifup=False`` they are composed with ``intp`` and
    act on the original chunk nodes.
    """

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
    targ = PointInfo.from_any(targobj)
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
    """Assemble one source-panel product-quadrature matrix.

    The output maps densities on the original ``chnkr.k`` source nodes to
    target values.  This mirrors the MATLAB ``chunkerkerneval_pquad`` block
    assembly, but stays local to one source panel.
    """

    k = chnkr.k
    if nodes is None or weights is None:
        t, w = lege.exps(2 * k)[:2]
    else:
        t = np.asarray(nodes, dtype=float).reshape(-1)
        w = np.asarray(weights, dtype=float).reshape(-1)
    interp = lege.matrin(k, t)[0]
    interp_ab = lege.matrin(k, np.array([-1.0, 1.0]))[0]
    targ = PointInfo.from_any(targobj)
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


def panel_matrix_auto_side(
    chnkr: Chunker,
    src_chunk: int,
    targobj: PointInfo | dict[str, Any] | ArrayLike,
    splitinfo: SplitInfo,
    *,
    side: str | None = None,
    nodes: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    side_tol: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble pquad blocks for targets whose interior/exterior side is known.

    If ``side`` is supplied, every target is evaluated on that side. Otherwise
    the side is inferred from the nearest source node normal on ``src_chunk``.
    Targets too close to the source panel to classify robustly are left
    unhandled so callers can use their Gauss fallback.
    """

    targ = PointInfo.from_any(targobj)
    op0, op1 = splitinfo.opdims
    ntarg = int(targ.r.shape[1])
    shape = (op0 * ntarg, op1 * chnkr.k)
    if ntarg == 0:
        return np.zeros(shape), np.zeros(0, dtype=bool)

    groups = _side_groups(chnkr, src_chunk, targ, side=side, side_tol=side_tol)
    handled = np.zeros(ntarg, dtype=bool)
    out: np.ndarray | None = None
    for side0, target_ids in groups:
        if target_ids.size == 0:
            continue
        block = panel_matrix(chnkr, src_chunk, _take_pointinfo(targ, target_ids), splitinfo, side0, nodes=nodes, weights=weights)
        if out is None:
            out = np.zeros(shape, dtype=block.dtype)
        elif np.result_type(out.dtype, block.dtype) != out.dtype:
            out = out.astype(np.result_type(out.dtype, block.dtype), copy=False)
        rows = _target_rows(target_ids, op0)
        out[rows, :] = block
        handled[target_ids] = True
    if out is None:
        out = np.zeros(shape)
    return np.real_if_close(out), handled


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
    tangent = dz_up / speed
    normal = -1j * tangent
    wxp = w_arr * dz_up

    special: tuple[np.ndarray, ...] = ()
    if nout:
        special = sd_special_quad(
            target[0] + 1j * target[1],
            z_up,
            normal,
            wxp,
            xlohi[0],
            xlohi[1],
            side0,
            nout=nout,
        )
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
        else:  # pragma: no cover - _required_special_count validates this.
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
    """Return Helsing-Ojala special weights ``As, A, A1, A2``.

    ``source_wxp`` is the complex speed-weight product ``w * z'(t)`` on the
    source rule nodes.
    """

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
    scale = params.get("_scale", 1.0)
    if name == "laplace":
        return _laplace_splitinfo(kind, params.get("coefs", None), scale)
    if name == "helmholtz":
        return _helmholtz_splitinfo(kind, params.get("zk", None), params.get("coefs", None), opdims, scale)
    return None


def _laplace_splitinfo(kind: str, coefs: Any, scale: Any = 1.0) -> SplitInfo | None:
    if kind in {"s", "single"}:
        return SplitInfo((LOG,), ("r",), lambda s, t: (scale * _ones(t, s),), (1, 1))
    if kind in {"d", "double"}:
        return SplitInfo((CAUCHY,), ("r",), lambda s, t: (scale * _ones(t, s),), (1, 1))
    if kind in {"c", "combined"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs).reshape(-1, order="F")

        def functions(s: PointInfo, t: PointInfo) -> tuple[np.ndarray, np.ndarray]:
            ones = _ones(t, s)
            return scale * c[1] * ones, scale * c[0] * ones

        return SplitInfo((LOG, CAUCHY), ("r", "r"), functions, (1, 1))
    return None


def _helmholtz_splitinfo(kind: str, zk: Any, coefs: Any, opdims: tuple[int, int], scale: Any = 1.0) -> SplitInfo | None:
    if zk is None or opdims != (1, 1):
        return None
    from chunkie.kernels import helmholtz as helm2d

    if kind in {"s", "single"}:
        return SplitInfo((SMOOTH, LOG), ("r", "r"), lambda s, t: _scale_split(scale, _helmholtz_s_split(helm2d, zk, s, t)), (1, 1))
    if kind in {"d", "double"}:
        return SplitInfo(
            (SMOOTH, LOG, CAUCHY),
            ("r", "r", "r"),
            lambda s, t: _scale_split(scale, _helmholtz_d_split(helm2d, zk, s, t)),
            (1, 1),
        )
    if kind in {"c", "combined"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs).reshape(-1, order="F")
        return SplitInfo(
            (SMOOTH, LOG, CAUCHY),
            ("r", "r", "r"),
            lambda s, t: _scale_split(scale, _helmholtz_c_split(helm2d, zk, c, s, t)),
            (1, 1),
        )
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


def _scale_split(scale: Any, values: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return tuple(scale * value for value in values)


def _side_groups(
    chnkr: Chunker,
    src_chunk: int,
    targ: PointInfo,
    *,
    side: str | None,
    side_tol: float | None,
) -> list[tuple[str, np.ndarray]]:
    explicit = _normalize_side(side)
    ntarg = int(targ.r.shape[1])
    if explicit is not None:
        return [(explicit, np.arange(ntarg, dtype=int))]

    if chnkr.dim != 2:
        return []
    tol = _default_side_tol(chnkr, src_chunk) if side_tol is None else float(side_tol)
    diff = targ.r[:, :, None] - chnkr.r[:, None, :, src_chunk]
    dist2 = np.sum(diff * diff, axis=0)
    nearest = np.argmin(dist2, axis=1)
    target_ids = np.arange(ntarg)
    offsets = targ.r[:, target_ids] - chnkr.r[:, nearest, src_chunk]
    normals = chnkr.n[:, nearest, src_chunk]
    signed = np.sum(offsets * normals, axis=0)
    inside = np.flatnonzero(signed < -tol)
    outside = np.flatnonzero(signed > tol)
    groups: list[tuple[str, np.ndarray]] = []
    if inside.size:
        groups.append(("i", inside.astype(int, copy=False)))
    if outside.size:
        groups.append(("e", outside.astype(int, copy=False)))
    return groups


def _normalize_side(side: str | None) -> str | None:
    if side is None:
        return None
    side0 = str(side).lower()
    if side0 not in {"i", "e"}:
        raise ValueError("side must be 'i' or 'e'")
    return side0


def _default_side_tol(chnkr: Chunker, src_chunk: int) -> float:
    try:
        scale = float(chnkr.chunklen()[src_chunk])
    except Exception:
        scale = 1.0
    return 1.0e-13 * max(1.0, scale)


def _take_pointinfo(info: PointInfo, indices: np.ndarray) -> PointInfo:
    return PointInfo(
        r=info.r[:, indices],
        d=None if info.d is None else info.d[:, indices],
        d2=None if info.d2 is None else info.d2[:, indices],
        n=None if info.n is None else info.n[:, indices],
        data=None if info.data is None else info.data[:, indices],
    )


def _target_rows(indices: np.ndarray, op0: int) -> np.ndarray:
    return (indices[:, None] * int(op0) + np.arange(int(op0))[None, :]).reshape(-1)
