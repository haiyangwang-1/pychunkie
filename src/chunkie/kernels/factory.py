"""Kernel wrappers for boundary-integral operator assembly.

The :func:`kernel` factory returns callable :class:`Kernel` objects with
operator dimensions, singularity metadata, and optional FMM evaluators. Kernel
objects can be added, subtracted, scaled, conjugated, or interleaved into block
systems; the metadata follows those algebraic operations so the operator layer
can choose dense, FMM, special-quadrature, or FLAM paths.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..acceleration.fmm_kernel import (
    _biharm2d_fmm,
    _conj_fmm,
    _derived_fmm,
    _direct_fmm,
    _elast2d_fmm,
    _helm2d_fmm,
    _interleave_dtype,
    _interleave_fmm,
    _interleave_indices,
    _lap2d_fmm,
    _scale_fmm,
    _stok2d_fmm,
    _sum_fmm,
    _target_count,
)
from ..geometry import PointInfo
from . import (
    biharmonic as biharm2d,
)
from . import (
    elasticity as elast2d,
)
from . import (
    helmholtz as helm2d,
)
from . import (
    helmholtz_1d as helm1d,
)
from . import (
    laplace as lap2d,
)
from . import (
    stokes as stok2d,
)

_KERNEL_PROBE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    FloatingPointError,
    NotImplementedError,
)


@dataclass
class Kernel:
    """Callable PDE layer kernel with assembly metadata.

    ``opdims=(m, n)`` means the kernel maps an ``n``-component source density
    at each boundary node to an ``m``-component target value. ``sing`` records
    the strongest source-target singularity used by special quadrature
    dispatch: common values are ``"smooth"``, ``"log"``, ``"pv"``, and
    ``"hs"``. ``fmm`` is a matrix-free evaluator accepting weighted source
    densities; it is ``None`` when no accelerated path is available.
    """

    name: str = "custom"
    type: str = "custom"
    eval: Callable[[Any, Any], np.ndarray] | None = None
    fmm: Callable[[float, Any, Any, np.ndarray], Any] | None = None
    opdims: tuple[int, int] = (0, 0)
    sing: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    isnan: bool = False
    iszero: bool = False

    def __call__(self, source: Any, target: Any) -> np.ndarray:
        if self.eval is None:
            raise ValueError("kernel has no evaluator")
        return self.eval(source, target)

    def __add__(self, other: Kernel) -> Kernel:
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
            iszero=bool(self.iszero and other.iszero),
        )

    def __sub__(self, other: Kernel) -> Kernel:
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
            iszero=bool(self.iszero and other.iszero),
        )

    def __neg__(self) -> Kernel:
        return self * -1.0

    def __mul__(self, scalar: float | complex) -> Kernel:
        if not np.isscalar(scalar):
            raise TypeError("kernel multiplication only supports scalars")
        if np.isnan(scalar):
            return nans(*self.opdims)
        if scalar == 0:
            return zeros(*self.opdims)
        params = self.params.copy()
        params["_scale"] = params.get("_scale", 1.0) * scalar
        scaled_fmm = None
        if self.fmm is not None:
            scaled_fmm = _derived_fmm(
                lambda eps, s, t, sigma: _scale_fmm(self.fmm(eps, s, t, sigma), scalar),
                self.fmm,
            )
        return Kernel(
            name=self.name,
            type=self.type,
            eval=lambda s, t: scalar * self(s, t),
            fmm=scaled_fmm,
            opdims=self.opdims,
            sing=self.sing,
            params=params,
            isnan=self.isnan,
            iszero=bool(self.iszero or scalar == 0),
        )

    def __rmul__(self, scalar: float | complex) -> Kernel:
        return self * scalar

    def __truediv__(self, scalar: float | complex) -> Kernel:
        if not np.isscalar(scalar):
            raise TypeError("kernel division only supports scalars")
        if np.isnan(scalar):
            return nans(*self.opdims)
        if scalar == 0:
            raise ZeroDivisionError("kernel division by zero")
        return self * (1.0 / scalar)

    def conj(self) -> Kernel:
        params = self.params.copy()
        if "_scale" in params:
            params["_scale"] = np.conj(params["_scale"])
        conj_fmm = None
        if self.fmm is not None:
            conj_fmm = _derived_fmm(
                lambda eps, s, t, sigma: _conj_fmm(self.fmm(eps, s, t, sigma)),
                self.fmm,
            )
        return Kernel(
            name=self.name,
            type=self.type,
            eval=lambda s, t: np.conj(self(s, t)),
            fmm=conj_fmm,
            opdims=self.opdims,
            sing=self.sing,
            params=params,
            isnan=self.isnan,
            iszero=self.iszero,
        )

    def conjugate(self) -> Kernel:
        return self.conj()

    @staticmethod
    def zeros(m: int = 1, n: int | None = None) -> Kernel:
        return zeros(m, n)

    @staticmethod
    def nans(m: int = 1, n: int | None = None) -> Kernel:
        return nans(m, n)


def kernel(spec: str | Callable[[Any, Any], np.ndarray] | Kernel, *args: Any) -> Kernel:
    """Build a kernel from a family name, callable, existing kernel, or block spec.

    String families include ``"lap"``/``"laplace"``, ``"helm"``/``"helmholtz"``,
    ``"helmdiff"``, ``"helm1d"``, ``"biharm"``, ``"stok"``/``"stokes"``,
    ``"elast"``, ``"zero"``, and ``"nan"``. A callable is wrapped as a custom
    dense kernel. A 2D list or object array of kernels is interleaved into a
    block kernel whose density and value components are stored node-by-node in
    Fortran order.
    """

    if isinstance(spec, Kernel):
        return spec
    if isinstance(spec, (list, tuple, np.ndarray)):
        return interleave(spec)
    if callable(spec):
        return Kernel(eval=spec, fmm=_direct_fmm(spec), opdims=_infer_opdims(spec))
    if not isinstance(spec, str):
        raise TypeError("kernel must be a name, callable, or Kernel")

    name = spec.lower()
    if name in {"laplace", "lap", "l"}:
        return lap2d_kernel(*args)
    if name in {"helmholtz", "helm", "h"}:
        return helm2d_kernel(*args)
    if name in {"helmholtz difference", "helmdiff", "hdiff", "helm_diff"}:
        return helm2ddiff_kernel(*args)
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
    raise ValueError(f"Kernel {spec!r} not found")


def lap2d_kernel(kind: str, coefs: Any | None = None) -> Kernel:
    """Build a 2D Laplace layer kernel."""

    typ = kind.lower()
    if typ in {"c", "combined"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return lap2d_kernel("d") * c[0] + lap2d_kernel("s") * c[1]
    if typ in {"cp", "cprime"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return lap2d_kernel("dp") * c[0] + lap2d_kernel("sp") * c[1]
    if typ in {"cg", "cgrad"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return lap2d_kernel("dg") * c[0] + lap2d_kernel("sg") * c[1]
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
        "dp": "hs",
        "dprime": "hs",
    }.get(typ, "log")
    return Kernel(
        name="laplace",
        type=typ,
        eval=lambda s, t: lap2d.kernel(s, t, typ, coefs),
        fmm=_lap2d_fmm(typ, coefs) or _direct_fmm(lambda s, t: lap2d.kernel(s, t, typ, coefs)),
        opdims=opdims,
        sing=sing,
        params={} if coefs is None else {"coefs": coefs},
    )


def helm2d_kernel(kind: str, zk: complex, coefs: Any | None = None) -> Kernel:
    """Build a 2D Helmholtz layer kernel for wavenumber ``zk``."""

    typ = kind.lower()
    if typ in {"all", "trans_sys", "ts", "trans_rep_grad", "trep_g", "trans_rep_g"}:
        opdims = (2, 2)
    elif typ in {"trans_rep", "trep", "trans_rep_prime", "trep_p", "trans_rep_p"}:
        opdims = (1, 2)
    elif typ in {"sg", "sgrad", "dg", "dgrad", "cg", "cgrad", "c2tr", "c2trans"}:
        opdims = (2, 1)
    else:
        opdims = (1, 1)
    if typ in {"c", "combined"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs)
        return helm2d_kernel("d", zk) * c[0] + helm2d_kernel("s", zk) * c[1]
    if typ in {"cp", "cprime"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs)
        return helm2d_kernel("dp", zk) * c[0] + helm2d_kernel("sp", zk) * c[1]
    if typ in {"cg", "cgrad"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs)
        return helm2d_kernel("dg", zk) * c[0] + helm2d_kernel("sg", zk) * c[1]
    sing = (
        "log"
        if typ in {"s", "single", "d", "double", "sp", "sprime", "trans_rep", "trep"}
        else "hs"
    )
    return Kernel(
        name="helmholtz",
        type=typ,
        eval=lambda s, t: helm2d.kernel(zk, s, t, typ, coefs),
        fmm=_helm2d_fmm(typ, zk, coefs)
        or _direct_fmm(lambda s, t: helm2d.kernel(zk, s, t, typ, coefs)),
        opdims=opdims,
        sing=sing,
        params={"zk": zk} if coefs is None else {"zk": zk, "coefs": coefs},
    )


def helm2ddiff_kernel(kind: str, zks: Any, coefs: Any | None = None) -> Kernel:
    """Build a Helmholtz-difference kernel with the Laplace singularity removed."""

    typ = kind.lower()
    z = np.asarray(zks).reshape(-1)
    if z.size != 2:
        raise ValueError("Helmholtz-difference kernels require exactly two wavenumbers")
    if typ in {"all", "trans_sys", "ts", "trans_rep_grad", "trep_g", "trans_rep_g"}:
        opdims = (2, 2)
    elif typ in {"trans_rep", "trep", "trans_rep_prime", "trep_p", "trans_rep_p"}:
        opdims = (1, 2)
    elif typ in {"sg", "sgrad", "dg", "dgrad", "cg", "cgrad", "c2tr", "c2trans"}:
        opdims = (2, 1)
    else:
        opdims = (1, 1)

    c = _helmdiff_default_coefs(typ) if coefs is None else np.asarray(coefs)

    def eval_(source_info: Any, target_info: Any) -> np.ndarray:
        if typ in {
            "c",
            "combined",
            "cp",
            "cprime",
            "trans_rep",
            "trep",
            "trans_rep_prime",
            "trep_p",
            "trans_rep_p",
        }:
            cmat = np.asarray(c)
            return helm2d.kernel(
                z[0], source_info, target_info, f"{typ}_diff", cmat[:, 0]
            ) - helm2d.kernel(z[1], source_info, target_info, f"{typ}_diff", cmat[:, 1])
        if typ in {"c2tr", "c2trans"}:
            carr = np.asarray(c)
            if carr.ndim == 2:
                return helm2d.kernel(
                    z[0], source_info, target_info, f"{typ}_diff", carr[:, 0]
                ) - helm2d.kernel(z[1], source_info, target_info, f"{typ}_diff", carr[:, 1])
            return helm2d.kernel(
                z[0], source_info, target_info, f"{typ}_diff", carr[:, :, 0]
            ) - helm2d.kernel(z[1], source_info, target_info, f"{typ}_diff", carr[:, :, 1])
        if typ in {"all", "trans_sys", "ts"}:
            carr = np.asarray(c)
            return helm2d.kernel(
                z[0], source_info, target_info, f"{typ}_diff", carr[:, :, 0]
            ) - helm2d.kernel(z[1], source_info, target_info, f"{typ}_diff", carr[:, :, 1])
        scale = np.asarray(c).reshape(-1)
        return scale[0] * helm2d.kernel(z[0], source_info, target_info, f"{typ}_diff") - scale[
            1
        ] * helm2d.kernel(z[1], source_info, target_info, f"{typ}_diff")

    return Kernel(
        name="helmholtz difference",
        type=typ,
        eval=eval_,
        fmm=_direct_fmm(eval_),
        opdims=opdims,
        sing="log",
        params={"zks": zks, "coefs": c},
    )


def _helmdiff_default_coefs(typ: str) -> np.ndarray:
    if typ in {"all", "trans_sys", "ts", "c2tr", "c2trans"}:
        return np.ones((2, 2, 2), dtype=float)
    if typ in {
        "c",
        "combined",
        "cp",
        "cprime",
        "trans_rep",
        "trep",
        "trans_rep_prime",
        "trep_p",
        "trans_rep_p",
    }:
        return np.ones((2, 2), dtype=float)
    return np.ones(2, dtype=float)


def helm1d_kernel(kind: str, zk: complex, coefs: Any | None = None) -> Kernel:
    """Build a 1D Helmholtz layer kernel for line-like point data."""

    typ = kind.lower()
    return Kernel(
        name="helmholtz1d",
        type=typ,
        eval=lambda s, t: helm1d.kernel(zk, s, t, typ, coefs),
        fmm=_direct_fmm(lambda s, t: helm1d.kernel(zk, s, t, typ, coefs)),
        opdims=(1, 1),
        sing="removable" if typ in {"s", "single"} else "smooth",
        params={"zk": zk} if coefs is None else {"zk": zk, "coefs": coefs},
    )


def biharm2d_kernel(kind: str) -> Kernel:
    """Build a 2D biharmonic Green-layer kernel."""

    typ = kind.lower()
    opdims = (2, 1) if typ in {"sgrad", "sg"} else (3, 1) if typ in {"shess", "hess"} else (1, 1)
    return Kernel(
        name="biharmonic",
        type=typ,
        eval=lambda s, t: biharm2d.kernel(s, t, typ),
        fmm=_biharm2d_fmm(typ) or _direct_fmm(lambda s, t: biharm2d.kernel(s, t, typ)),
        opdims=opdims,
        sing="log" if typ in {"s", "single", "lap", "slap", "laplacian"} else "pv",
    )


def stok2d_kernel(kind: str, mu: float = 1.0, coefs: Any | None = None) -> Kernel:
    """Build a 2D Stokes velocity, pressure, traction, or gradient kernel."""

    typ = kind.lower()
    opdims = (
        (1, 2)
        if typ in {"spres", "spressure", "dpres", "dpressure", "cpres", "cpressure"}
        else (4, 2)
        if typ in {"sg", "sgrad", "dg", "dgrad", "cg", "cgrad"}
        else (2, 2)
    )
    sing = {
        "s": "log",
        "single": "log",
        "svel": "log",
        "svelocity": "log",
        "spres": "pv",
        "spressure": "pv",
        "strac": "smooth",
        "straction": "smooth",
        "sgrad": "pv",
        "sg": "pv",
        "d": "smooth",
        "double": "smooth",
        "dvel": "smooth",
        "dvelocity": "smooth",
        "dpres": "hs",
        "dpressure": "hs",
        "dtrac": "hs",
        "dtraction": "hs",
        "dgrad": "hs",
        "dg": "hs",
        "c": "log",
        "combined": "log",
        "cvel": "log",
        "cvelocity": "log",
        "cpres": "hs",
        "cpressure": "hs",
        "ctrac": "hs",
        "ctraction": "hs",
        "cgrad": "hs",
        "cg": "hs",
    }.get(typ, "smooth")
    return Kernel(
        name="stokes",
        type=typ,
        eval=lambda s, t: stok2d.kernel(mu, s, t, typ, coefs),
        fmm=_stok2d_fmm(typ, mu, coefs)
        or _direct_fmm(lambda s, t: stok2d.kernel(mu, s, t, typ, coefs)),
        opdims=opdims,
        sing=sing,
        params={"mu": mu} if coefs is None else {"mu": mu, "coefs": coefs},
    )


def elast2d_kernel(kind: str, lam: float, mu: float) -> Kernel:
    """Build a 2D linear-elasticity kernel for Lame parameters ``lam`` and ``mu``."""

    typ = kind.lower()
    opdims = (4, 2) if typ in {"sgrad", "sg", "daltgrad", "daltg"} else (2, 2)
    sing = {
        "s": "log",
        "single": "log",
        "sgrad": "pv",
        "sg": "pv",
        "strac": "pv",
        "straction": "pv",
        "d": "pv",
        "double": "pv",
        "dalt": "smooth",
        "dalttrac": "hs",
        "dalttraction": "hs",
        "daltgrad": "hs",
        "daltg": "hs",
    }.get(typ, "smooth")
    return Kernel(
        name="elasticity",
        type=typ,
        eval=lambda s, t: elast2d.kernel(lam, mu, s, t, typ),
        fmm=_elast2d_fmm(typ, lam, mu)
        or _direct_fmm(lambda s, t: elast2d.kernel(lam, mu, s, t, typ)),
        opdims=opdims,
        sing=sing,
        params={"lam": lam, "mu": mu},
    )


def zeros(m: int = 1, n: int | None = None) -> Kernel:
    """Return an ``m`` by ``n`` zero block kernel."""

    n = m if n is None else n

    def eval_(source: Any, target: Any) -> np.ndarray:
        source_info = PointInfo.from_any(source)
        target_info = PointInfo.from_any(target)
        return np.zeros((m * target_info.r.shape[1], n * source_info.r.shape[1]))

    return Kernel(
        name="zeros",
        type="zeros",
        eval=eval_,
        fmm=lambda eps, s, t, sigma: np.zeros(m * _target_count(t)),
        opdims=(m, n),
        sing="smooth",
        iszero=True,
    )


def nans(m: int = 1, n: int | None = None) -> Kernel:
    """Return an ``m`` by ``n`` NaN block kernel for diagnostics/composition."""

    n = m if n is None else n

    def eval_(source: Any, target: Any) -> np.ndarray:
        source_info = PointInfo.from_any(source)
        target_info = PointInfo.from_any(target)
        return np.full((m * target_info.r.shape[1], n * source_info.r.shape[1]), np.nan)

    return Kernel(
        name="nans",
        type="nans",
        eval=eval_,
        fmm=lambda eps, s, t, sigma: np.full(m * _target_count(t), np.nan),
        opdims=(m, n),
        sing="smooth",
        isnan=True,
    )


def interleave(kernels: Any) -> Kernel:
    """Interleave a rectangular array of kernels into one block kernel.

    The resulting kernel stores block rows and columns node-interleaved:
    ``[u1(node1), u2(node1), u1(node2), ...]``. This matches the density layout
    expected by vector PDE kernels and chunkgraph edge-by-edge block systems.
    """

    arr = np.asarray(kernels, dtype=object)
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

    def eval_(source: Any, target: Any) -> np.ndarray:
        source_info = PointInfo.from_any(source)
        target_info = PointInfo.from_any(target)
        out = np.zeros(
            (opdims[0] * target_info.r.shape[1], opdims[1] * source_info.r.shape[1]),
            dtype=_interleave_dtype(items, source_info, target_info),
        )
        for i in range(items.shape[0]):
            ridx = _interleave_indices(target_info.r.shape[1], opdims[0], rowstarts[i], rowdims[i])
            for j in range(items.shape[1]):
                cidx = _interleave_indices(
                    source_info.r.shape[1], opdims[1], colstarts[j], coldims[j]
                )
                out[np.ix_(ridx, cidx)] = items[i, j](source_info, target_info)
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


def _infer_opdims(func: Callable[[Any, Any], np.ndarray]) -> tuple[int, int]:
    try:
        src = PointInfo(
            r=np.zeros((2, 1)), d=np.ones((2, 1)), d2=np.zeros((2, 1)), n=np.ones((2, 1))
        )
        targ = PointInfo(
            r=np.ones((2, 1)), d=np.ones((2, 1)), d2=np.zeros((2, 1)), n=np.ones((2, 1))
        )
        shape = func(src, targ).shape
        return int(shape[0]), int(shape[1])
    except _KERNEL_PROBE_EXCEPTIONS:
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
