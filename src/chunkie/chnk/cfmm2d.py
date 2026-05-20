"""Complex Cauchy-kernel FMM point kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from chunkie.operators import PointInfo, pointinfo


def kern(
    srcinfo: PointInfo | dict | ArrayLike,
    targinfo: PointInfo | dict | ArrayLike,
    kind: str,
) -> np.ndarray:
    """Evaluate dense matrices matching the current ``fmm2dpy.cfmm2d`` API.

    Source strengths are node-interleaved as ``(charge, dipole)`` for the
    two-component selectors. Coincident source-target terms are omitted.
    """

    src = pointinfo(srcinfo)
    targ = pointinfo(targinfo)
    typ = kind.lower()
    diff = _complex_points(targ)[:, None] - _complex_points(src)[None, :]
    nt, ns = diff.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        charge_pot = np.log(np.abs(diff))
        dipole_pot = 1.0 / diff
        charge_der = 1.0 / diff
        dipole_der = -1.0 / diff**2
        charge_hess = -1.0 / diff**2
        dipole_hess = 2.0 / diff**3

    charge_pot = _zero_nonfinite(charge_pot)
    dipole_pot = _zero_nonfinite(dipole_pot)
    charge_der = _zero_nonfinite(charge_der)
    dipole_der = _zero_nonfinite(dipole_der)
    charge_hess = _zero_nonfinite(charge_hess)
    dipole_hess = _zero_nonfinite(dipole_hess)

    if typ in {"c", "charge", "charges", "log"}:
        return charge_pot
    if typ in {"d", "dipole", "dipoles", "cauchy"}:
        return dipole_pot
    if typ in {"p", "pot", "potential"}:
        return _interleave_1x2(charge_pot, dipole_pot, nt, ns)
    if typ in {"g", "grad", "der", "derivative", "dz"}:
        return _interleave_1x2(charge_der, dipole_der, nt, ns)
    if typ in {"h", "hess", "second", "second_derivative", "dzz"}:
        return _interleave_1x2(charge_hess, dipole_hess, nt, ns)
    if typ in {"all", "pgh"}:
        out = np.zeros((3 * nt, 2 * ns), dtype=complex)
        out[0::3] = _interleave_1x2(charge_pot, dipole_pot, nt, ns)
        out[1::3] = _interleave_1x2(charge_der, dipole_der, nt, ns)
        out[2::3] = _interleave_1x2(charge_hess, dipole_hess, nt, ns)
        return out
    raise ValueError(f"Unknown cfmm2d kernel type {kind!r}.")


def _interleave_1x2(left: np.ndarray, right: np.ndarray, nt: int, ns: int) -> np.ndarray:
    out = np.zeros((nt, 2 * ns), dtype=np.result_type(left, right))
    out[:, 0::2] = left
    out[:, 1::2] = right
    return out


def _complex_points(info: PointInfo) -> np.ndarray:
    return np.asarray(info.r[0] + 1j * info.r[1], dtype=np.complex128).reshape(-1)


def _zero_nonfinite(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, 0.0)
