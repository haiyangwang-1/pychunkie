"""Small kernel wrapper compatible with dense direct operators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chnk import biharm2d, elast2d, helm1d, helm2d, lap2d, stok2d

try:  # pragma: no cover - exercised when the optional compiled package imports.
    import fmm2dpy as _fmm2dpy
except Exception:  # pragma: no cover - keep source installs usable without FMM2D.
    _fmm2dpy = None


@dataclass
class Kernel:
    name: str = "custom"
    type: str = "custom"
    eval: Callable[[Any, Any], np.ndarray] | None = None
    fmm: Callable[[float, Any, Any, np.ndarray], Any] | None = None
    opdims: tuple[int, int] = (0, 0)
    sing: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    isnan: bool = False
    iszero: bool = False

    def __call__(self, srcinfo: Any, targinfo: Any) -> np.ndarray:
        if self.eval is None:
            raise ValueError("kernel has no evaluator")
        return self.eval(srcinfo, targinfo)

    def __add__(self, other: "Kernel") -> "Kernel":
        other = kernel(other)
        if self.opdims != other.opdims:
            raise ValueError("kernel dimensions must agree to add")
        if self.isnan or other.isnan:
            return nans(*self.opdims)
        return Kernel(
            name=f"custom {self.name} {other.name}",
            type="sum",
            eval=lambda s, t: self(s, t) + other(s, t),
            fmm=_sum_fmm(self, other, 1.0),
            opdims=self.opdims,
            sing=_worst_sing(self.sing, other.sing),
            iszero=self.iszero and other.iszero,
        )

    def __sub__(self, other: "Kernel") -> "Kernel":
        other = kernel(other)
        if self.opdims != other.opdims:
            raise ValueError("kernel dimensions must agree to subtract")
        if self.isnan or other.isnan:
            return nans(*self.opdims)
        return Kernel(
            name=f"custom {self.name} {other.name}",
            type="difference",
            eval=lambda s, t: self(s, t) - other(s, t),
            fmm=_sum_fmm(self, other, -1.0),
            opdims=self.opdims,
            sing=_worst_sing(self.sing, other.sing),
            iszero=self.iszero and other.iszero,
        )

    def __neg__(self) -> "Kernel":
        return self * -1.0

    def __mul__(self, scalar: float | complex) -> "Kernel":
        if not np.isscalar(scalar):
            raise TypeError("kernel multiplication only supports scalars")
        if np.isnan(scalar):
            return nans(*self.opdims)
        return Kernel(
            name=self.name,
            type=self.type,
            eval=lambda s, t: scalar * self(s, t),
            fmm=None if self.fmm is None else lambda eps, s, t, sigma: _scale_fmm(self.fmm(eps, s, t, sigma), scalar),
            opdims=self.opdims,
            sing=self.sing,
            params=self.params.copy(),
            isnan=self.isnan,
            iszero=self.iszero or scalar == 0,
        )

    def __rmul__(self, scalar: float | complex) -> "Kernel":
        return self * scalar

    def __truediv__(self, scalar: float | complex) -> "Kernel":
        if not np.isscalar(scalar):
            raise TypeError("kernel division only supports scalars")
        if np.isnan(scalar):
            return nans(*self.opdims)
        if scalar == 0:
            raise ZeroDivisionError("kernel division by zero")
        return self * (1.0 / scalar)

    def conj(self) -> "Kernel":
        return Kernel(
            name=self.name,
            type=self.type,
            eval=lambda s, t: np.conj(self(s, t)),
            fmm=None if self.fmm is None else lambda eps, s, t, sigma: _conj_fmm(self.fmm(eps, s, t, sigma)),
            opdims=self.opdims,
            sing=self.sing,
            params=self.params.copy(),
            isnan=self.isnan,
            iszero=self.iszero,
        )

    def conjugate(self) -> "Kernel":
        return self.conj()

    @staticmethod
    def zeros(m: int = 1, n: int | None = None) -> "Kernel":
        return zeros(m, n)

    @staticmethod
    def nans(m: int = 1, n: int | None = None) -> "Kernel":
        return nans(m, n)


def kernel(kern: str | Callable[[Any, Any], np.ndarray] | Kernel, *args: Any) -> Kernel:
    """MATLAB-style kernel constructor."""

    if isinstance(kern, Kernel):
        return kern
    if isinstance(kern, (list, tuple, np.ndarray)):
        return interleave(kern)
    if callable(kern):
        return Kernel(eval=kern, fmm=_direct_fmm(kern), opdims=_infer_opdims(kern))
    if not isinstance(kern, str):
        raise TypeError("kernel must be a name, callable, or Kernel")

    name = kern.lower()
    if name in {"laplace", "lap", "l"}:
        return lap2d_kernel(*args)
    if name in {"helmholtz", "helm", "h"}:
        return helm2d_kernel(*args)
    if name in {"helmholtz1d", "helm1d", "h1d"}:
        return helm1d_kernel(*args)
    if name in {"biharmonic", "biharm", "b"}:
        return biharm2d_kernel(*args)
    if name in {"stokes", "stok"}:
        return stok2d_kernel(*args)
    if name in {"elasticity", "elast", "e"}:
        return elast2d_kernel(*args)
    if name in {"zeros", "zero", "z"}:
        return zeros(*args)
    if name in {"nans", "nan"}:
        return nans(*args)
    raise ValueError(f"Kernel {kern!r} not found")


def lap2d_kernel(kind: str, coefs: Any | None = None) -> Kernel:
    typ = kind.lower()
    opdims = (2, 1) if typ in {"sg", "sgrad", "dg", "dgrad"} else (1, 1)
    sing = {
        "s": "log",
        "single": "log",
        "d": "smooth",
        "double": "smooth",
        "sp": "smooth",
        "sprime": "smooth",
        "st": "pv",
        "stau": "pv",
        "sg": "pv",
        "sgrad": "pv",
        "dg": "hs",
        "dgrad": "hs",
    }.get(typ, "log")
    if typ in {"c", "combined"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return lap2d_kernel("d") * c[0] + lap2d_kernel("s") * c[1]
    return Kernel(
        name="laplace",
        type=typ,
        eval=lambda s, t: lap2d.kern(s, t, typ, coefs),
        fmm=_lap2d_fmm(typ, coefs) or _direct_fmm(lambda s, t: lap2d.kern(s, t, typ, coefs)),
        opdims=opdims,
        sing=sing,
        params={} if coefs is None else {"coefs": coefs},
    )


def helm2d_kernel(kind: str, zk: complex, coefs: Any | None = None) -> Kernel:
    typ = kind.lower()
    opdims = (2, 1) if typ in {"sg", "sgrad", "dg", "dgrad"} else (1, 1)
    if typ in {"c", "combined"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs)
        return helm2d_kernel("d", zk) * c[0] + helm2d_kernel("s", zk) * c[1]
    return Kernel(
        name="helmholtz",
        type=typ,
        eval=lambda s, t: helm2d.kern(zk, s, t, typ, coefs),
        fmm=_helm2d_fmm(typ, zk, coefs) or _direct_fmm(lambda s, t: helm2d.kern(zk, s, t, typ, coefs)),
        opdims=opdims,
        sing="log" if typ in {"s", "single", "d", "double", "sp", "sprime"} else "hs",
        params={"zk": zk} if coefs is None else {"zk": zk, "coefs": coefs},
    )


def helm1d_kernel(kind: str, zk: complex, coefs: Any | None = None) -> Kernel:
    typ = kind.lower()
    return Kernel(
        name="helmholtz1d",
        type=typ,
        eval=lambda s, t: helm1d.kern(zk, s, t, typ, coefs),
        fmm=_direct_fmm(lambda s, t: helm1d.kern(zk, s, t, typ, coefs)),
        opdims=(1, 1),
        sing="removable" if typ in {"s", "single"} else "smooth",
        params={"zk": zk} if coefs is None else {"zk": zk, "coefs": coefs},
    )


def biharm2d_kernel(kind: str) -> Kernel:
    typ = kind.lower()
    opdims = (2, 1) if typ in {"sgrad", "sg"} else (3, 1) if typ in {"shess", "hess"} else (1, 1)
    return Kernel(
        name="biharmonic",
        type=typ,
        eval=lambda s, t: biharm2d.kern(s, t, typ),
        fmm=_direct_fmm(lambda s, t: biharm2d.kern(s, t, typ)),
        opdims=opdims,
        sing="log" if typ in {"s", "single", "lap", "slap", "laplacian"} else "pv",
    )


def stok2d_kernel(kind: str, mu: float = 1.0, coefs: Any | None = None) -> Kernel:
    typ = kind.lower()
    opdims = (1, 2) if typ in {"spres", "spressure", "dpres", "dpressure", "cpres", "cpressure"} else (4, 2) if typ in {"sg", "sgrad", "dg", "dgrad", "cg", "cgrad"} else (2, 2)
    return Kernel(
        name="stokes",
        type=typ,
        eval=lambda s, t: stok2d.kern(mu, s, t, typ, coefs),
        fmm=_stok2d_fmm(typ, mu, coefs) or _direct_fmm(lambda s, t: stok2d.kern(mu, s, t, typ, coefs)),
        opdims=opdims,
        sing="log" if typ in {"s", "single", "svel", "svelocity", "c", "combined", "cvel", "cvelocity"} else "smooth",
        params={"mu": mu} if coefs is None else {"mu": mu, "coefs": coefs},
    )


def elast2d_kernel(kind: str, lam: float, mu: float) -> Kernel:
    typ = kind.lower()
    opdims = (4, 2) if typ in {"sgrad", "sg", "daltgrad", "daltg"} else (2, 2)
    return Kernel(
        name="elasticity",
        type=typ,
        eval=lambda s, t: elast2d.kern(lam, mu, s, t, typ),
        fmm=_direct_fmm(lambda s, t: elast2d.kern(lam, mu, s, t, typ)),
        opdims=opdims,
        sing="log" if typ in {"s", "single"} else "pv" if typ in {"d", "double", "strac"} else "smooth",
        params={"lam": lam, "mu": mu},
    )


def zeros(m: int = 1, n: int | None = None) -> Kernel:
    n = m if n is None else n
    return Kernel(
        name="zeros",
        type="zeros",
        eval=lambda s, t: np.zeros((m * t.r.shape[1], n * s.r.shape[1])),
        fmm=lambda eps, s, t, sigma: np.zeros(m * _target_count(t)),
        opdims=(m, n),
        sing="smooth",
        iszero=True,
    )


def nans(m: int = 1, n: int | None = None) -> Kernel:
    n = m if n is None else n
    return Kernel(
        name="nans",
        type="nans",
        eval=lambda s, t: np.full((m * t.r.shape[1], n * s.r.shape[1]), np.nan),
        fmm=lambda eps, s, t, sigma: np.full(m * _target_count(t), np.nan),
        opdims=(m, n),
        sing="smooth",
        isnan=True,
    )


def interleave(kerns: Any) -> Kernel:
    arr = np.asarray(kerns, dtype=object)
    if arr.ndim == 0:
        return kernel(arr.item())
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("kernel interleave expects a 2D array-like")

    items = np.empty(arr.shape, dtype=object)
    for idx in np.ndindex(arr.shape):
        items[idx] = kernel(arr[idx])
        if items[idx].isnan:
            raise ValueError("kernel interleave does not support nan kernels")

    rowdims = [int(items[i, 0].opdims[0]) for i in range(items.shape[0])]
    coldims = [int(items[0, j].opdims[1]) for j in range(items.shape[1])]
    for i in range(items.shape[0]):
        for j in range(items.shape[1]):
            if int(items[i, j].opdims[0]) != rowdims[i] or int(items[i, j].opdims[1]) != coldims[j]:
                raise ValueError("kernel block opdims are inconsistent")

    opdims = (sum(rowdims), sum(coldims))
    rowstarts = np.concatenate(([0], np.cumsum(rowdims)))
    colstarts = np.concatenate(([0], np.cumsum(coldims)))

    def eval_(srcinfo: Any, targinfo: Any) -> np.ndarray:
        from .operators import pointinfo

        src = pointinfo(srcinfo)
        targ = pointinfo(targinfo)
        out = np.zeros((opdims[0] * targ.r.shape[1], opdims[1] * src.r.shape[1]), dtype=_interleave_dtype(items, src, targ))
        for i in range(items.shape[0]):
            ridx = _interleave_indices(targ.r.shape[1], opdims[0], rowstarts[i], rowdims[i])
            for j in range(items.shape[1]):
                cidx = _interleave_indices(src.r.shape[1], opdims[1], colstarts[j], coldims[j])
                out[np.ix_(ridx, cidx)] = items[i, j](src, targ)
        return out

    fmm = _interleave_fmm(items, opdims, rowstarts, colstarts, rowdims, coldims)
    return Kernel(
        name="interleave",
        type="interleave",
        eval=eval_,
        fmm=fmm,
        opdims=opdims,
        sing=_worst_many([items[idx].sing for idx in np.ndindex(items.shape)]),
        params={"blocks": items.tolist()},
        iszero=all(items[idx].iszero for idx in np.ndindex(items.shape)),
    )


def _direct_fmm(func: Callable[[Any, Any], np.ndarray]) -> Callable[[float, Any, Any, np.ndarray], np.ndarray]:
    def fmm_eval(eps: float, srcinfo: Any, targinfo: Any, sigma: np.ndarray) -> np.ndarray:
        _ = eps
        from .operators import pointinfo

        src = pointinfo(srcinfo)
        targ = pointinfo(targinfo)
        return func(src, targ) @ np.asarray(sigma).reshape(-1, order="F")

    return fmm_eval


def _lap2d_fmm(kind: str, coefs: Any | None = None) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if _fmm2dpy is None:
        return None
    typ = kind.lower()
    if typ in {"c", "combined"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_lap2d_fmm("d"), _lap2d_fmm("s"), c[0], c[1])
    if typ not in {"s", "single", "d", "double", "sgrad", "sg", "dgrad", "dg"}:
        return None

    def fmm_eval(eps: float, srcinfo: Any, targinfo: Any, sigma: np.ndarray) -> np.ndarray:
        from .operators import pointinfo

        src = pointinfo(srcinfo)
        targ = pointinfo(targinfo)
        sig = np.asarray(sigma).reshape(-1, order="F")
        if typ in {"s", "single", "sgrad", "sg"}:
            out = _fmm2dpy.lfmm2d(eps=eps, sources=src.r, charges=sig, targets=targ.r, pgt=2)
        else:
            if src.n is None:
                raise ValueError("source normals are required")
            out = _fmm2dpy.lfmm2d(eps=eps, sources=src.r, dipstr=sig, dipvec=src.n, targets=targ.r, pgt=3)
        scale = -1.0 / (2.0 * np.pi)
        if typ in {"s", "single", "d", "double"}:
            return np.real_if_close(scale * np.asarray(out.pottarg).reshape(-1, order="F"))
        grad = np.asarray(out.gradtarg)
        return np.real_if_close(scale * grad.reshape(-1, order="F"))

    return fmm_eval


def _helm2d_fmm(kind: str, zk: complex, coefs: Any | None = None) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if _fmm2dpy is None:
        return None
    typ = kind.lower()
    if typ in {"c", "combined"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_helm2d_fmm("d", zk), _helm2d_fmm("s", zk), c[0], c[1])
    if typ not in {"s", "single", "d", "double", "sgrad", "sg", "dgrad", "dg"}:
        return None

    def fmm_eval(eps: float, srcinfo: Any, targinfo: Any, sigma: np.ndarray) -> np.ndarray:
        from .operators import pointinfo

        src = pointinfo(srcinfo)
        targ = pointinfo(targinfo)
        sig = np.asarray(sigma).reshape(-1, order="F")
        if typ in {"s", "single", "sgrad", "sg"}:
            out = _fmm2dpy.hfmm2d(eps=eps, zk=zk, sources=src.r, charges=sig, targets=targ.r, pgt=2)
        else:
            if src.n is None:
                raise ValueError("source normals are required")
            pgt = 2 if typ in {"dgrad", "dg"} else 1
            out = _fmm2dpy.hfmm2d(eps=eps, zk=zk, sources=src.r, dipstr=sig, dipvec=src.n, targets=targ.r, pgt=pgt)
        if typ in {"s", "single", "d", "double"}:
            return np.asarray(out.pottarg).reshape(-1, order="F")
        grad = np.asarray(out.gradtarg)
        return grad.reshape(-1, order="F")

    return fmm_eval


def _stok2d_fmm(kind: str, mu: float = 1.0, coefs: Any | None = None) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if _fmm2dpy is None:
        return None
    typ = kind.lower()
    if typ in {"c", "combined", "cvel", "cvelocity"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_stok2d_fmm("d", mu), _stok2d_fmm("s", mu), c[0], c[1])
    if typ in {"cpres", "cpressure"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_stok2d_fmm("dpres", mu), _stok2d_fmm("spres", mu), c[0], c[1])
    if typ in {"cg", "cgrad"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_stok2d_fmm("dgrad", mu), _stok2d_fmm("sgrad", mu), c[0], c[1])
    if typ not in {
        "s",
        "single",
        "svel",
        "svelocity",
        "d",
        "double",
        "dvel",
        "dvelocity",
        "spres",
        "spressure",
        "dpres",
        "dpressure",
        "sgrad",
        "sg",
        "dgrad",
        "dg",
    }:
        return None

    def fmm_eval(eps: float, srcinfo: Any, targinfo: Any, sigma: np.ndarray) -> np.ndarray:
        from .operators import pointinfo

        src = pointinfo(srcinfo)
        targ = pointinfo(targinfo)
        sig = np.asarray(sigma).reshape(2, -1, order="F")
        is_double = typ in {"d", "double", "dvel", "dvelocity", "dpres", "dpressure", "dgrad", "dg"}
        if is_double and src.n is None:
            raise ValueError("source normals are required")
        ifppregtarg = 3 if typ in {"sgrad", "sg", "dgrad", "dg"} else 2 if typ in {"spres", "spressure", "dpres", "dpressure"} else 1
        kwargs = {"strslet": sig, "strsvec": src.n} if is_double else {"stoklet": sig}
        out = _fmm2dpy.stfmm2d(eps=eps, sources=src.r, targets=targ.r, ifppregtarg=ifppregtarg, **kwargs)

        if typ in {"s", "single", "svel", "svelocity"}:
            scale = 1.0 / (2.0 * np.pi * float(mu))
            return scale * np.asarray(out.pottarg)[0].reshape(-1, order="F")
        if typ in {"d", "double", "dvel", "dvelocity"}:
            return -1.0 / (2.0 * np.pi) * np.asarray(out.pottarg)[0].reshape(-1, order="F")
        if typ in {"spres", "spressure"}:
            return 1.0 / (2.0 * np.pi) * np.asarray(out.pretarg)[0].reshape(-1, order="F")
        if typ in {"dpres", "dpressure"}:
            return -float(mu) / (2.0 * np.pi) * np.asarray(out.pretarg)[0].reshape(-1, order="F")
        if typ in {"sgrad", "sg"}:
            scale = 1.0 / (2.0 * np.pi * float(mu))
            return scale * np.asarray(out.gradtarg)[0].reshape(-1, order="F")
        return -1.0 / (2.0 * np.pi) * np.asarray(out.gradtarg)[0].reshape(-1, order="F")

    return fmm_eval


def _sum_raw_fmm(
    left: Callable[[float, Any, Any, np.ndarray], np.ndarray] | None,
    right: Callable[[float, Any, Any, np.ndarray], np.ndarray] | None,
    left_scale: float | complex,
    right_scale: float | complex,
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if left is None or right is None:
        return None

    def fmm_eval(eps: float, srcinfo: Any, targinfo: Any, sigma: np.ndarray) -> np.ndarray:
        return left_scale * left(eps, srcinfo, targinfo, sigma) + right_scale * right(eps, srcinfo, targinfo, sigma)

    return fmm_eval


def _target_count(targinfo: Any) -> int:
    from .operators import pointinfo

    return pointinfo(targinfo).r.shape[1]


def _sum_fmm(left: Kernel, right: Kernel, sign: float) -> Callable[[float, Any, Any, np.ndarray], Any] | None:
    if left.fmm is None or right.fmm is None:
        return None

    def fmm_eval(eps: float, srcinfo: Any, targinfo: Any, sigma: np.ndarray) -> Any:
        return _add_fmm(left.fmm(eps, srcinfo, targinfo, sigma), right.fmm(eps, srcinfo, targinfo, sigma), sign)

    return fmm_eval


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


def _interleave_dtype(items: np.ndarray, src: Any, targ: Any) -> np.dtype:
    dtype = np.dtype(float)
    for item in items.flat:
        try:
            dtype = np.result_type(dtype, np.asarray(item(src, targ)).dtype)
        except Exception:
            pass
    return dtype


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

    def fmm_eval(eps: float, srcinfo: Any, targinfo: Any, sigma: np.ndarray) -> np.ndarray:
        from .operators import pointinfo

        src = pointinfo(srcinfo)
        targ = pointinfo(targinfo)
        sig = np.asarray(sigma).reshape(-1, order="F")
        out = np.zeros(opdims[0] * targ.r.shape[1], dtype=np.result_type(sig, complex if any(np.iscomplexobj(item.params) for item in items.flat) else float))
        for i in range(items.shape[0]):
            ridx = _interleave_indices(targ.r.shape[1], opdims[0], rowstarts[i], rowdims[i])
            accum = np.zeros(ridx.size, dtype=out.dtype)
            for j in range(items.shape[1]):
                cidx = _interleave_indices(src.r.shape[1], opdims[1], colstarts[j], coldims[j])
                vals = items[i, j].fmm(eps, src, targ, sig[cidx])
                if isinstance(vals, tuple):
                    vals = vals[0]
                accum = accum + np.asarray(vals).reshape(-1, order="F")
            out[ridx] = accum
        return out

    return fmm_eval


def _infer_opdims(func: Callable[[Any, Any], np.ndarray]) -> tuple[int, int]:
    try:
        from .operators import PointInfo

        src = PointInfo(r=np.zeros((2, 1)), d=np.ones((2, 1)), d2=np.zeros((2, 1)), n=np.ones((2, 1)))
        targ = PointInfo(r=np.ones((2, 1)), d=np.ones((2, 1)), d2=np.zeros((2, 1)), n=np.ones((2, 1)))
        shape = func(src, targ).shape
        return int(shape[0]), int(shape[1])
    except Exception:
        return (0, 0)


def _worst_sing(a: str, b: str) -> str:
    order = {"": 0, "smooth": 1, "log": 2, "pv": 3, "hs": 4}
    reverse = {value: key for key, value in order.items()}
    return reverse[max(order.get(a, 0), order.get(b, 0))]


def _worst_many(sings: list[str]) -> str:
    out = "smooth"
    for sing in sings:
        out = _worst_sing(out, sing)
    return out
