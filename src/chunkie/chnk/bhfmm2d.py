"""Current ``fmm2dpy.bhfmm2d`` complex biharmonic point kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from chunkie.operators import PointInfo, pointinfo


def kern(
    srcinfo: PointInfo | dict | ArrayLike,
    targinfo: PointInfo | dict | ArrayLike,
    kind: str,
) -> np.ndarray:
    """Evaluate dense matrices matching the current ``fmm2dpy.bhfmm2d`` API.

    Source strengths are node-interleaved as
    ``(c1, c2, v1, v2, v3)``. Coincident source-target terms are omitted.
    """

    src = pointinfo(srcinfo)
    targ = pointinfo(targinfo)
    typ = kind.lower()
    diff = _complex_points(targ)[:, None] - _complex_points(src)[None, :]
    cdiff = np.conjugate(diff)
    nt, ns = diff.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        pot_c1 = 2.0 * np.log(np.abs(diff))
        pot_c2 = diff / cdiff
        pot_v1 = 1.0 / diff
        pot_v2 = diff / cdiff**2
        pot_v3 = 1.0 / cdiff

        gz_c1 = 1.0 / diff
        gz_v1 = -1.0 / diff**2
        gzc_c2 = 1.0 / cdiff
        gzc_v2 = 1.0 / cdiff**2
        gc_c1 = 1.0 / cdiff
        gc_c2 = -diff / cdiff**2
        gc_v2 = -2.0 * diff / cdiff**3
        gc_v3 = -1.0 / cdiff**2

    pot = [_zero_nonfinite(item) for item in (pot_c1, pot_c2, pot_v1, pot_v2, pot_v3)]
    gz_c1 = _zero_nonfinite(gz_c1)
    gz_v1 = _zero_nonfinite(gz_v1)
    gzc_c2 = _zero_nonfinite(gzc_c2)
    gzc_v2 = _zero_nonfinite(gzc_v2)
    gc_c1 = _zero_nonfinite(gc_c1)
    gc_c2 = _zero_nonfinite(gc_c2)
    gc_v2 = _zero_nonfinite(gc_v2)
    gc_v3 = _zero_nonfinite(gc_v3)

    if typ in {"p", "pot", "potential"}:
        return _interleave_potential(pot, nt, ns)
    if typ in {"g", "grad", "gradient", "derivative"}:
        return _interleave_gradient(gz_c1, gz_v1, gzc_c2, gzc_v2, gc_c1, gc_c2, gc_v2, gc_v3, nt, ns)
    if typ in {"all", "pg"}:
        grad = _interleave_gradient(gz_c1, gz_v1, gzc_c2, gzc_v2, gc_c1, gc_c2, gc_v2, gc_v3, nt, ns)
        out = np.zeros((4 * nt, 5 * ns), dtype=complex)
        out[0::4] = _interleave_potential(pot, nt, ns)
        out[1::4] = grad[0::3]
        out[2::4] = grad[1::3]
        out[3::4] = grad[2::3]
        return out
    raise ValueError(f"Unknown bhfmm2d kernel type {kind!r}.")


def _interleave_potential(blocks: list[np.ndarray], nt: int, ns: int) -> np.ndarray:
    out = np.zeros((nt, 5 * ns), dtype=complex)
    for idx, block in enumerate(blocks):
        out[:, idx::5] = block
    return out


def _interleave_gradient(
    gz_c1: np.ndarray,
    gz_v1: np.ndarray,
    gzc_c2: np.ndarray,
    gzc_v2: np.ndarray,
    gc_c1: np.ndarray,
    gc_c2: np.ndarray,
    gc_v2: np.ndarray,
    gc_v3: np.ndarray,
    nt: int,
    ns: int,
) -> np.ndarray:
    out = np.zeros((3 * nt, 5 * ns), dtype=complex)
    out[0::3, 0::5] = gz_c1
    out[0::3, 2::5] = gz_v1
    out[1::3, 1::5] = gzc_c2
    out[1::3, 3::5] = gzc_v2
    out[2::3, 0::5] = gc_c1
    out[2::3, 1::5] = gc_c2
    out[2::3, 3::5] = gc_v2
    out[2::3, 4::5] = gc_v3
    return out


def _complex_points(info: PointInfo) -> np.ndarray:
    return np.asarray(info.r[0] + 1j * info.r[1], dtype=np.complex128).reshape(-1)


def _zero_nonfinite(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, 0.0)
