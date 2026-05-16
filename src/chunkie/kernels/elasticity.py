"""Two-dimensional elasticity kernel placeholders."""

from __future__ import annotations

from .base import Kernel


def kernel(selector: str = "s", *, lame_lambda: float, lame_mu: float) -> Kernel:
    raise NotImplementedError("Elasticity kernels are a required rewrite milestone after scalar kernels")
