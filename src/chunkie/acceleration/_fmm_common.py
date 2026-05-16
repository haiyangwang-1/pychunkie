"""Common FMM evaluator composition helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from chunkie._layout import as_boundary_vector
from chunkie.geometry import PointInfo

from .fmm import fmm2dpy

_DIRECT_FMM_FALLBACK_ATTR = "_chunkie_direct_fmm_fallback"
_KERNEL_PROBE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    FloatingPointError,
    NotImplementedError,
)


def _direct_fmm(
    func: Callable[[Any, Any], np.ndarray],
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray]:
    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        _ = eps
        src = PointInfo.from_any(source_info)
        targ = PointInfo.from_any(target_info)
        return func(src, targ) @ as_boundary_vector(sigma, name="density")

    return _mark_direct_fmm_fallback(fmm_eval)


def _mark_direct_fmm_fallback(
    fmm: Callable[[float, Any, Any, np.ndarray], Any],
) -> Callable[[float, Any, Any, np.ndarray], Any]:
    setattr(fmm, _DIRECT_FMM_FALLBACK_ATTR, True)
    return fmm


def _is_direct_fmm_fallback(fmm: Callable[[float, Any, Any, np.ndarray], Any] | None) -> bool:
    return bool(getattr(fmm, _DIRECT_FMM_FALLBACK_ATTR, False))


def _derived_fmm(
    fmm: Callable[[float, Any, Any, np.ndarray], Any],
    *parents: Callable[[float, Any, Any, np.ndarray], Any] | None,
) -> Callable[[float, Any, Any, np.ndarray], Any]:
    if any(_is_direct_fmm_fallback(parent) for parent in parents):
        _mark_direct_fmm_fallback(fmm)
    return fmm


def _laplace_log_moments(
    eps: float,
    source: Any,
    target: Any,
    charges: np.ndarray,
    pgt: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    charge_arr = np.asarray(charges)
    if charge_arr.ndim == 1:
        charge_arr = charge_arr.reshape(1, -1)
    nd = int(charge_arr.shape[0])
    nt = int(target.r.shape[1])
    out = fmm2dpy.lfmm2d(
        eps=eps,
        sources=source.r,
        charges=charge_arr if nd > 1 else charge_arr[0],
        targets=target.r,
        pgt=pgt,
        nd=nd,
    )
    pot = np.asarray(out.pottarg).reshape(nd, nt, order="C")
    grad = None
    hess = None
    if pgt >= 2:
        grad = np.asarray(out.gradtarg).reshape(nd, 2, nt, order="C")
    if pgt >= 3:
        hess = np.asarray(out.hesstarg).reshape(nd, 3, nt, order="C")
    return pot, grad, hess


def _sum_raw_fmm(
    left: Callable[[float, Any, Any, np.ndarray], np.ndarray] | None,
    right: Callable[[float, Any, Any, np.ndarray], np.ndarray] | None,
    left_scale: float | complex,
    right_scale: float | complex,
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if left is None or right is None:
        return None

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        return left_scale * left(eps, source_info, target_info, sigma) + right_scale * right(
            eps, source_info, target_info, sigma
        )

    return _derived_fmm(fmm_eval, left, right)


def _target_count(target_info: Any) -> int:
    return PointInfo.from_any(target_info).r.shape[1]


def _sum_fmm(left: Any, right: Any, sign: float) -> Callable[[float, Any, Any, np.ndarray], Any] | None:
    if left.fmm is None or right.fmm is None:
        return None

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> Any:
        return _add_fmm(
            left.fmm(eps, source_info, target_info, sigma),
            right.fmm(eps, source_info, target_info, sigma),
            sign,
        )

    return _derived_fmm(fmm_eval, left.fmm, right.fmm)


def _add_fmm(left: Any, right: Any, sign: float) -> Any:
    if isinstance(left, tuple) or isinstance(right, tuple):
        lt = left if isinstance(left, tuple) else (left,)
        rt = right if isinstance(right, tuple) else (right,)
        nout = max(len(lt), len(rt))
        out = []
        for i in range(nout):
            li = lt[i] if i < len(lt) else 0.0
            ri = rt[i] if i < len(rt) else 0.0
            out.append(li + sign * ri)
        return tuple(out)
    return left + sign * right


def _scale_fmm(value: Any, scalar: float | complex) -> Any:
    if isinstance(value, tuple):
        return tuple(scalar * item for item in value)
    return scalar * value


def _conj_fmm(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(np.conj(item) for item in value)
    return np.conj(value)


def _interleave_indices(npt: int, total_dim: int, offset: int, dim: int) -> np.ndarray:
    base = np.arange(npt)[:, None] * total_dim + offset
    return (base + np.arange(dim)[None, :]).reshape(-1)


def _interleave_dtype(items: np.ndarray, source: Any, target: Any) -> np.dtype:
    dtype = np.dtype(float)
    for item in items.flat:
        probed = _probe_kernel_dtype(item, source, target)
        if probed is not None:
            dtype = np.result_type(dtype, probed)
    return dtype


def _probe_kernel_dtype(
    item: Callable[[Any, Any], np.ndarray], source: Any, target: Any
) -> np.dtype | None:
    try:
        return np.asarray(item(source, target)).dtype
    except _KERNEL_PROBE_EXCEPTIONS:
        return None


def _interleave_fmm(
    items: np.ndarray,
    opdims: tuple[int, int],
    rowstarts: np.ndarray,
    colstarts: np.ndarray,
    rowdims: list[int],
    coldims: list[int],
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if any(item.fmm is None for item in items.flat):
        return None

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        src = PointInfo.from_any(source_info)
        targ = PointInfo.from_any(target_info)
        sig = as_boundary_vector(sigma, name="density")
        has_complex_params = any(_contains_complex_value(item.params) for item in items.flat)
        out = np.zeros(
            opdims[0] * targ.r.shape[1],
            dtype=np.result_type(sig, complex if has_complex_params else float),
        )
        for i in range(items.shape[0]):
            ridx = _interleave_indices(targ.r.shape[1], opdims[0], rowstarts[i], rowdims[i])
            accum = np.zeros(ridx.size, dtype=out.dtype)
            for j in range(items.shape[1]):
                cidx = _interleave_indices(src.r.shape[1], opdims[1], colstarts[j], coldims[j])
                vals = items[i, j].fmm(eps, src, targ, sig[cidx])
                if isinstance(vals, tuple):
                    vals = vals[0]
                accum = accum + as_boundary_vector(vals, name="interleaved FMM values")
            if np.result_type(out.dtype, accum.dtype) != out.dtype:
                out = out.astype(np.result_type(out.dtype, accum.dtype), copy=False)
            out[ridx] = accum
        return out

    return _derived_fmm(fmm_eval, *(item.fmm for item in items.flat))


def _contains_complex_value(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_complex_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_complex_value(item) for item in value)
    try:
        return bool(np.iscomplexobj(value))
    except TypeError:
        return False
