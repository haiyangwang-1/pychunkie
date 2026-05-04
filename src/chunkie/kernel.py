"""Small kernel wrapper compatible with dense direct operators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chnk import elast2d, helm1d, helm2d, lap2d, stok2d


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
        fmm=_direct_fmm(lambda s, t: lap2d.kern(s, t, typ, coefs)),
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
        fmm=_direct_fmm(lambda s, t: helm2d.kern(zk, s, t, typ, coefs)),
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


def stok2d_kernel(kind: str, mu: float = 1.0, coefs: Any | None = None) -> Kernel:
    typ = kind.lower()
    opdims = (1, 2) if typ in {"spres", "spressure", "dpres", "dpressure", "cpres", "cpressure"} else (4, 2) if typ in {"sg", "sgrad", "dg", "dgrad", "cg", "cgrad"} else (2, 2)
    return Kernel(
        name="stokes",
        type=typ,
        eval=lambda s, t: stok2d.kern(mu, s, t, typ, coefs),
        fmm=_direct_fmm(lambda s, t: stok2d.kern(mu, s, t, typ, coefs)),
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


def _direct_fmm(func: Callable[[Any, Any], np.ndarray]) -> Callable[[float, Any, Any, np.ndarray], np.ndarray]:
    def fmm_eval(eps: float, srcinfo: Any, targinfo: Any, sigma: np.ndarray) -> np.ndarray:
        _ = eps
        from .operators import pointinfo

        src = pointinfo(srcinfo)
        targ = pointinfo(targinfo)
        return func(src, targ) @ np.asarray(sigma).reshape(-1, order="F")

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
