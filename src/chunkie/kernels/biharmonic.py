"""Two-dimensional biharmonic kernel placeholders."""

from __future__ import annotations

from .base import Kernel


def kernel(selector: str = "s") -> Kernel:
    raise NotImplementedError("Biharmonic kernels are a required rewrite milestone after Laplace/Helmholtz")
