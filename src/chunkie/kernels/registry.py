"""Kernel factory registry."""

from __future__ import annotations

from .base import Kernel


def kernel(spec: str | Kernel = "laplace", *, selector: str = "s", **params) -> Kernel:
    if isinstance(spec, Kernel):
        return spec
    family = spec.lower()
    if family in {"laplace", "lap2d"}:
        from . import laplace

        return laplace.kernel(selector)
    if family in {"helmholtz", "helm2d"}:
        from . import helmholtz

        if "wavenumber" not in params:
            raise ValueError("Helmholtz kernels require wavenumber=<complex>")
        return helmholtz.kernel(selector, wavenumber=params["wavenumber"])
    if family == "stokes":
        from . import stokes

        return stokes.kernel(selector, viscosity=params.get("viscosity", 1.0))
    if family == "biharmonic":
        from . import biharmonic

        return biharmonic.kernel(selector)
    if family == "elasticity":
        from . import elasticity

        return elasticity.kernel(
            selector,
            lame_lambda=params["lame_lambda"],
            lame_mu=params["lame_mu"],
        )
    raise ValueError(f"unknown kernel family {spec!r}")
