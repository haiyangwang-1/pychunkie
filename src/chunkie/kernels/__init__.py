"""Concrete physics kernel families used by :mod:`chunkie.kernel`."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SUBMODULES = {
    "biharmonic",
    "elasticity",
    "helmholtz",
    "helmholtz_1d",
    "laplace",
    "stokes",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "biharmonic",
    "elasticity",
    "helmholtz",
    "helmholtz_1d",
    "laplace",
    "stokes",
]
