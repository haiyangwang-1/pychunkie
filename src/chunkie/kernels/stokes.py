"""Two-dimensional Stokes kernel placeholders."""

from __future__ import annotations

from .base import Kernel


def kernel(selector: str = "s", *, viscosity: float = 1.0) -> Kernel:
    raise NotImplementedError("Stokes kernels are a required rewrite milestone after scalar kernels")
