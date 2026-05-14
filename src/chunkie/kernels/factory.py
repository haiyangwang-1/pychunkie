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

from chunkie._layout import as_boundary_field_matrix, as_boundary_vector

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

try:  # pragma: no cover - exercised when the optional compiled package imports.
    import fmm2dpy as _fmm2dpy
except (ImportError, OSError):  # pragma: no cover - keep source installs usable without FMM2D.
    _fmm2dpy = None


_KERNEL_PROBE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    FloatingPointError,
    NotImplementedError,
)


_DIRECT_FMM_FALLBACK_ATTR = "_chunkie_direct_fmm_fallback"


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


def _lap2d_fmm(
    kind: str, coefs: Any | None = None
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if _fmm2dpy is None:
        return None
    typ = kind.lower()
    if typ in {"c", "combined"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_lap2d_fmm("d"), _lap2d_fmm("s"), c[0], c[1])
    if typ in {"cp", "cprime"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_lap2d_fmm("dp"), _lap2d_fmm("sp"), c[0], c[1])
    if typ in {"cg", "cgrad"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_lap2d_fmm("dg"), _lap2d_fmm("sg"), c[0], c[1])
    if typ not in {
        "s",
        "single",
        "d",
        "double",
        "sp",
        "sprime",
        "st",
        "stau",
        "hilb",
        "sgrad",
        "sg",
        "dgrad",
        "dg",
        "dp",
        "dprime",
    }:
        return None

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        src = PointInfo.from_any(source_info)
        targ = PointInfo.from_any(target_info)
        sig = as_boundary_vector(sigma, name="density")
        if typ in {"s", "single", "sgrad", "sg", "sp", "sprime", "st", "stau"}:
            out = _fmm2dpy.lfmm2d(eps=eps, sources=src.r, charges=sig, targets=targ.r, pgt=2)
        elif typ in {"hilb"}:
            if src.n is None:
                raise ValueError("source normals are required")
            dipvec = np.vstack((-src.n[1], src.n[0]))
            out = _fmm2dpy.lfmm2d(
                eps=eps, sources=src.r, dipstr=2.0 * sig, dipvec=dipvec, targets=targ.r, pgt=1
            )
        else:
            if src.n is None:
                raise ValueError("source normals are required")
            out = _fmm2dpy.lfmm2d(
                eps=eps, sources=src.r, dipstr=sig, dipvec=src.n, targets=targ.r, pgt=3
            )
        scale = -1.0 / (2.0 * np.pi)
        if typ in {"s", "single", "d", "double", "hilb"}:
            return np.real_if_close(
                scale * as_boundary_vector(out.pottarg, name="Laplace FMM potential")
            )
        grad = scale * np.asarray(out.gradtarg)
        if typ in {"sgrad", "sg", "dgrad", "dg"}:
            return np.real_if_close(as_boundary_vector(grad, name="Laplace FMM gradient"))
        if typ in {"sp", "sprime"}:
            if targ.n is None:
                raise ValueError("target normals are required")
            return np.real_if_close(grad[0] * targ.n[0] + grad[1] * targ.n[1])
        if typ in {"st", "stau"}:
            if targ.n is None:
                raise ValueError("target normals are required")
            return np.real_if_close(-grad[0] * targ.n[1] + grad[1] * targ.n[0])
        if typ in {"dp", "dprime"}:
            if targ.n is None:
                raise ValueError("target normals are required")
            return np.real_if_close(grad[0] * targ.n[0] + grad[1] * targ.n[1])
        raise ValueError(f"Unknown Laplace FMM selector {kind!r}")

    return fmm_eval


def _biharm2d_fmm(kind: str) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    typ = kind.lower()
    if _fmm2dpy is None:
        return None
    if typ not in {
        "s",
        "single",
        "d",
        "double",
        "sp",
        "sprime",
        "sgrad",
        "sg",
        "shess",
        "hess",
        "lap",
        "slap",
        "laplacian",
    }:
        return None

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        src = PointInfo.from_any(source_info)
        targ = PointInfo.from_any(target_info)
        sig = as_boundary_vector(sigma, name="density")
        sx = src.r[0]
        sy = src.r[1]
        tx = targ.r[0]
        ty = targ.r[1]
        nt = targ.r.shape[1]

        if typ in {"d", "double"}:
            if src.n is None:
                raise ValueError("source normals are required")
            nx = src.n[0]
            ny = src.n[1]
            charges = np.vstack((sig * nx, sig * ny, sig * (nx * sx + ny * sy)))
            pot, _, _ = _laplace_log_moments(eps, src, targ, charges, pgt=1)
            moment = tx * pot[0] + ty * pot[1] - pot[2]
            affine = (
                tx * np.sum(sig * nx) + ty * np.sum(sig * ny) - np.sum(sig * (nx * sx + ny * sy))
            )
            return -(2.0 * moment + affine) / (8.0 * np.pi)

        if typ in {"sp", "sprime"}:
            if targ.n is None:
                raise ValueError("target normals are required")
            grad = as_boundary_field_matrix(
                _biharm2d_fmm("sgrad")(eps, src, targ, sig),
                2,
                nt,
                name="biharmonic gradient",
            )
            return grad[0] * targ.n[0] + grad[1] * targ.n[1]

        if typ in {"lap", "slap", "laplacian"}:
            pot, _, _ = _laplace_log_moments(eps, src, targ, sig, pgt=2)
            return (pot[0] + np.sum(sig)) / (2.0 * np.pi)

        charges = np.vstack((sig, sig * sx, sig * sy, sig * (sx * sx + sy * sy)))
        pot, grad, _ = _laplace_log_moments(eps, src, targ, charges, pgt=2)
        p0, px, py, p2 = pot
        g0, gx, gy, g2 = grad
        t2 = tx * tx + ty * ty

        if typ in {"s", "single"}:
            vals = t2 * p0 - 2.0 * tx * px - 2.0 * ty * py + p2
            return vals / (8.0 * np.pi)

        if typ in {"sgrad", "sg"}:
            out_x = (
                2.0 * tx * p0 + t2 * g0[0] - 2.0 * px - 2.0 * tx * gx[0] - 2.0 * ty * gy[0] + g2[0]
            )
            out_y = (
                2.0 * ty * p0 + t2 * g0[1] - 2.0 * tx * gx[1] - 2.0 * py - 2.0 * ty * gy[1] + g2[1]
            )
            return as_boundary_vector(
                np.vstack((out_x, out_y)) / (8.0 * np.pi),
                name="biharmonic gradient",
            )

        rx2 = tx * g0[0] - gx[0]
        rxy = ty * g0[0] - gy[0]
        ry2 = ty * g0[1] - gy[1]
        total = np.sum(sig)
        hxx = 2.0 * p0 + total + 2.0 * rx2
        hxy = 2.0 * rxy
        hyy = 2.0 * p0 + total + 2.0 * ry2
        if typ in {"shess", "hess"}:
            return as_boundary_vector(
                np.vstack((hxx, hxy, hyy)) / (8.0 * np.pi),
                name="biharmonic hessian",
            )
        return (hxx + hyy) / (8.0 * np.pi)

    return fmm_eval


def _elast2d_fmm(
    kind: str, lam: float, mu: float
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if _fmm2dpy is None:
        return None
    typ = kind.lower()
    if typ not in {
        "s",
        "single",
        "strac",
        "sgrad",
        "sg",
        "d",
        "double",
        "dalt",
        "daltgrad",
        "daltg",
        "dalttrac",
    }:
        return None
    beta = (lam + 3.0 * mu) / (4.0 * np.pi * mu * (lam + 2.0 * mu))
    gamma = -(lam + mu) / (4.0 * np.pi * mu * (lam + 2.0 * mu))
    eta = mu / (2.0 * np.pi * (lam + 2.0 * mu))
    zeta = (lam + mu) / (np.pi * (lam + 2.0 * mu))
    lap_d = _lap2d_fmm("d")
    lap_dgrad = _lap2d_fmm("dgrad")
    stok_d = _stok2d_fmm("d", mu)
    stok_dgrad = _stok2d_fmm("dgrad", mu)

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        src = PointInfo.from_any(source_info)
        targ = PointInfo.from_any(target_info)
        sig = as_boundary_field_matrix(
            sigma, 2, np.asarray(sigma).size // 2, name="elasticity density"
        )
        nt = targ.r.shape[1]
        if typ in {"s", "single", "sgrad", "sg", "strac"}:
            grad = typ in {"sgrad", "sg", "strac"}
            vals = _elast_single_fmm(eps, src, targ, sig, beta, gamma, want_grad=grad)
            if typ in {"s", "single"}:
                return vals
            grad_vals = as_boundary_field_matrix(vals, 4, nt, name="elasticity gradient")
            if typ in {"sgrad", "sg"}:
                return vals
            if targ.n is None:
                raise ValueError("target normals are required")
            return as_boundary_vector(
                _elastic_traction_from_grad(grad_vals, targ.n, lam, mu),
                name="elasticity traction",
            )

        if src.n is None:
            raise ValueError("source normals are required")
        if typ in {"d", "double"}:
            if lap_d is None or stok_d is None:
                raise ValueError(
                    "Laplace and Stokes FMM backends are required for elasticity double-layer FMM"
                )
            vel = (
                -zeta
                * np.pi
                * as_boundary_field_matrix(
                    stok_d(eps, src, targ, as_boundary_vector(sig)),
                    2,
                    nt,
                    name="Stokes velocity",
                )
            )
            norm = as_boundary_vector(lap_d(eps, src, targ, sig[0]), name="normal component")
            rot_x = as_boundary_vector(
                lap_d(eps, _with_normals(src, np.vstack((-src.n[1], src.n[0]))), targ, sig[1]),
                name="rotated x component",
            )
            rot_y = as_boundary_vector(
                lap_d(eps, _with_normals(src, np.vstack((src.n[1], -src.n[0]))), targ, sig[0]),
                name="rotated y component",
            )
            norm_y = as_boundary_vector(lap_d(eps, src, targ, sig[1]), name="normal y component")
            vel[0] += -eta * (2.0 * np.pi) * (norm + rot_x)
            vel[1] += -eta * (2.0 * np.pi) * (rot_y + norm_y)
            return as_boundary_vector(vel, name="elasticity velocity")

        if typ in {"dalt", "daltgrad", "daltg", "dalttrac"}:
            if (
                lap_d is None
                or stok_d is None
                or (typ not in {"dalt"} and (lap_dgrad is None or stok_dgrad is None))
            ):
                raise ValueError(
                    "Laplace and Stokes FMM backends are required for elasticity alternate double-layer FMM"
                )
            if typ == "dalt":
                vel = (
                    -zeta
                    * np.pi
                    * as_boundary_field_matrix(
                        stok_d(eps, src, targ, as_boundary_vector(sig)),
                        2,
                        nt,
                        name="Stokes velocity",
                    )
                )
                vel[0] += -4.0 * np.pi * eta * as_boundary_vector(lap_d(eps, src, targ, sig[0]))
                vel[1] += -4.0 * np.pi * eta * as_boundary_vector(lap_d(eps, src, targ, sig[1]))
                return as_boundary_vector(vel, name="elasticity velocity")

            grad_vals = (
                -zeta
                * np.pi
                * as_boundary_field_matrix(
                    stok_dgrad(eps, src, targ, as_boundary_vector(sig)),
                    4,
                    nt,
                    name="Stokes gradient",
                )
            )
            gx = as_boundary_field_matrix(lap_dgrad(eps, src, targ, sig[0]), 2, nt)
            gy = as_boundary_field_matrix(lap_dgrad(eps, src, targ, sig[1]), 2, nt)
            grad_vals[0] += -4.0 * np.pi * eta * gx[0]
            grad_vals[1] += -4.0 * np.pi * eta * gx[1]
            grad_vals[2] += -4.0 * np.pi * eta * gy[0]
            grad_vals[3] += -4.0 * np.pi * eta * gy[1]
            if typ in {"daltgrad", "daltg"}:
                return as_boundary_vector(grad_vals, name="elasticity gradient")
            if targ.n is None:
                raise ValueError("target normals are required")
            return as_boundary_vector(
                _elastic_traction_from_grad(grad_vals, targ.n, lam, mu),
                name="elasticity traction",
            )

        raise ValueError(f"Unknown elasticity FMM selector {kind!r}")

    return fmm_eval


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
    out = _fmm2dpy.lfmm2d(
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


def _elast_single_fmm(
    eps: float,
    source: Any,
    target: Any,
    sig: np.ndarray,
    beta: float,
    gamma: float,
    *,
    want_grad: bool,
) -> np.ndarray:
    sx = source.r[0]
    sy = source.r[1]
    tx = target.r[0]
    ty = target.r[1]
    a = sig[0]
    b = sig[1]
    charges = np.vstack((a, b, a * sx, a * sy, b * sx, b * sy))
    pot, grad, hess = _laplace_log_moments(eps, source, target, charges, pgt=3 if want_grad else 2)
    pa, pb = pot[0], pot[1]
    ga, gb, gasx, gasy, gbsx, gbsy = grad

    axx_a = tx * ga[0] - gasx[0]
    axy_a = ty * ga[0] - gasy[0]
    axy_b = ty * gb[0] - gbsy[0]
    ayy_b = ty * gb[1] - gbsy[1]

    ux = beta * pa + 0.5 * gamma * np.sum(a) + gamma * (axx_a + axy_b)
    uy = beta * pb + 0.5 * gamma * np.sum(b) + gamma * (axy_a + ayy_b)

    if not want_grad:
        return as_boundary_vector(np.vstack((ux, uy)), name="elasticity single-layer velocity")

    ha, hb, hasx, hasy, hbsx, hbsy = hess
    dux_dx = beta * ga[0] + gamma * ((ga[0] + tx * ha[0] - hasx[0]) + (ty * hb[0] - hbsy[0]))
    dux_dy = beta * ga[1] + gamma * ((tx * ha[1] - hasx[1]) + (gb[0] + ty * hb[1] - hbsy[1]))
    duy_dx = beta * gb[0] + gamma * ((ty * ha[0] - hasy[0]) + (ty * hb[1] - hbsy[1]))
    duy_dy = beta * gb[1] + gamma * (
        (ga[0] + ty * ha[1] - hasy[1]) + (gb[1] + ty * hb[2] - hbsy[2])
    )
    return as_boundary_vector(
        np.vstack((dux_dx, dux_dy, duy_dx, duy_dy)),
        name="elasticity single-layer gradient",
    )


def _elastic_traction_from_grad(
    grad: np.ndarray, normals: np.ndarray, lam: float, mu: float
) -> np.ndarray:
    nx = normals[0]
    ny = normals[1]
    du11, du12, du21, du22 = grad
    div = du11 + du22
    shear = float(mu) * (du12 + du21)
    tx = float(lam) * nx * div + shear * ny + 2.0 * float(mu) * du11 * nx
    ty = float(lam) * ny * div + shear * nx + 2.0 * float(mu) * du22 * ny
    return np.vstack((tx, ty))


def _with_normals(source: Any, normals: np.ndarray) -> Any:
    base = PointInfo.from_any(source)
    return PointInfo(r=base.r, d=base.d, d2=base.d2, n=np.asarray(normals), data=base.data)


def _helm2d_fmm(
    kind: str, zk: complex, coefs: Any | None = None
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if _fmm2dpy is None:
        return None
    typ = kind.lower()
    if typ in {"c", "combined"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_helm2d_fmm("d", zk), _helm2d_fmm("s", zk), c[0], c[1])
    if typ in {"cp", "cprime"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_helm2d_fmm("dp", zk), _helm2d_fmm("sp", zk), c[0], c[1])
    if typ not in {
        "s",
        "single",
        "d",
        "double",
        "sp",
        "sprime",
        "stau",
        "st",
        "sgrad",
        "sg",
        "dgrad",
        "dg",
        "dp",
        "dprime",
    }:
        return None

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        src = PointInfo.from_any(source_info)
        targ = PointInfo.from_any(target_info)
        sig = as_boundary_vector(sigma, name="density")
        if typ in {"s", "single", "sgrad", "sg", "sp", "sprime", "stau", "st"}:
            out = _fmm2dpy.hfmm2d(eps=eps, zk=zk, sources=src.r, charges=sig, targets=targ.r, pgt=2)
        else:
            if src.n is None:
                raise ValueError("source normals are required")
            pgt = 2 if typ in {"dgrad", "dg", "dp", "dprime"} else 1
            out = _fmm2dpy.hfmm2d(
                eps=eps, zk=zk, sources=src.r, dipstr=sig, dipvec=src.n, targets=targ.r, pgt=pgt
            )
        if typ in {"s", "single", "d", "double"}:
            return as_boundary_vector(out.pottarg, name="Helmholtz FMM potential")
        grad = np.asarray(out.gradtarg)
        if typ in {"sgrad", "sg", "dgrad", "dg"}:
            return as_boundary_vector(grad, name="Helmholtz FMM gradient")
        if typ in {"sp", "sprime", "dp", "dprime"}:
            if targ.n is None:
                raise ValueError("target normals are required")
            return grad[0] * targ.n[0] + grad[1] * targ.n[1]
        if typ in {"stau", "st"}:
            if targ.d is None:
                raise ValueError("target tangents are required")
            speed = np.sqrt(targ.d[0] ** 2 + targ.d[1] ** 2)
            return (grad[0] * targ.d[0] + grad[1] * targ.d[1]) / speed
        raise ValueError(f"Unknown Helmholtz FMM selector {kind!r}")

    return fmm_eval


def _stok2d_fmm(
    kind: str, mu: float = 1.0, coefs: Any | None = None
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if _fmm2dpy is None:
        return None
    typ = kind.lower()
    if typ in {"c", "combined", "cvel", "cvelocity"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_stok2d_fmm("d", mu), _stok2d_fmm("s", mu), c[0], c[1])
    if typ in {"cpres", "cpressure"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_stok2d_fmm("dpres", mu), _stok2d_fmm("spres", mu), c[0], c[1])
    if typ in {"ctrac", "ctraction"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_stok2d_fmm("dtrac", mu), _stok2d_fmm("strac", mu), c[0], c[1])
    if typ in {"cg", "cgrad"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return _sum_raw_fmm(_stok2d_fmm("dgrad", mu), _stok2d_fmm("sgrad", mu), c[0], c[1])
    if typ in {"strac", "straction"}:
        return _stok2d_traction_fmm(_stok2d_fmm("spres", mu), _stok2d_fmm("sgrad", mu), mu)
    if typ in {"dtrac", "dtraction"}:
        return _stok2d_traction_fmm(_stok2d_fmm("dpres", mu), _stok2d_fmm("dgrad", mu), mu)
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

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        src = PointInfo.from_any(source_info)
        targ = PointInfo.from_any(target_info)
        sig = as_boundary_field_matrix(sigma, 2, np.asarray(sigma).size // 2, name="Stokes density")
        is_double = typ in {"d", "double", "dvel", "dvelocity", "dpres", "dpressure", "dgrad", "dg"}
        if is_double and src.n is None:
            raise ValueError("source normals are required")
        ifppregtarg = (
            3
            if typ in {"sgrad", "sg", "dgrad", "dg"}
            else 2
            if typ in {"spres", "spressure", "dpres", "dpressure"}
            else 1
        )
        kwargs = {"strslet": sig, "strsvec": src.n} if is_double else {"stoklet": sig}
        out = _fmm2dpy.stfmm2d(
            eps=eps, sources=src.r, targets=targ.r, ifppregtarg=ifppregtarg, **kwargs
        )

        if typ in {"s", "single", "svel", "svelocity"}:
            scale = 1.0 / (2.0 * np.pi * float(mu))
            return scale * as_boundary_vector(np.asarray(out.pottarg)[0], name="Stokes velocity")
        if typ in {"d", "double", "dvel", "dvelocity"}:
            return (
                -1.0
                / (2.0 * np.pi)
                * as_boundary_vector(np.asarray(out.pottarg)[0], name="Stokes velocity")
            )
        if typ in {"spres", "spressure"}:
            return (
                1.0
                / (2.0 * np.pi)
                * as_boundary_vector(np.asarray(out.pretarg)[0], name="Stokes pressure")
            )
        if typ in {"dpres", "dpressure"}:
            return (
                -float(mu)
                / (2.0 * np.pi)
                * as_boundary_vector(np.asarray(out.pretarg)[0], name="Stokes pressure")
            )
        if typ in {"sgrad", "sg"}:
            scale = 1.0 / (2.0 * np.pi * float(mu))
            return scale * as_boundary_vector(np.asarray(out.gradtarg)[0], name="Stokes gradient")
        return (
            -1.0
            / (2.0 * np.pi)
            * as_boundary_vector(np.asarray(out.gradtarg)[0], name="Stokes gradient")
        )

    return fmm_eval


def _stok2d_traction_fmm(
    pressure_fmm: Callable[[float, Any, Any, np.ndarray], np.ndarray] | None,
    grad_fmm: Callable[[float, Any, Any, np.ndarray], np.ndarray] | None,
    mu: float,
) -> Callable[[float, Any, Any, np.ndarray], np.ndarray] | None:
    if pressure_fmm is None or grad_fmm is None:
        return None

    def fmm_eval(eps: float, source_info: Any, target_info: Any, sigma: np.ndarray) -> np.ndarray:
        targ = PointInfo.from_any(target_info)
        if targ.n is None:
            raise ValueError("target normals are required")
        pressure = as_boundary_vector(
            pressure_fmm(eps, source_info, targ, sigma), name="Stokes pressure"
        )
        grad = as_boundary_field_matrix(
            grad_fmm(eps, source_info, targ, sigma),
            4,
            targ.r.shape[1],
            name="Stokes gradient",
        )
        nx = targ.n[0]
        ny = targ.n[1]
        du11, du12, du21, du22 = grad
        mut = float(mu)
        tx = -pressure * nx + mut * (2.0 * du11 * nx + (du12 + du21) * ny)
        ty = -pressure * ny + mut * ((du21 + du12) * nx + 2.0 * du22 * ny)
        return as_boundary_vector(np.vstack((tx, ty)), name="Stokes traction")

    return fmm_eval


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


def _sum_fmm(
    left: Kernel, right: Kernel, sign: float
) -> Callable[[float, Any, Any, np.ndarray], Any] | None:
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
