"""Kernel factory/algebra and concrete physics kernel families."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SUBMODULES = {
    "biharmonic",
    "elasticity",
    "factory",
    "helmholtz",
    "helmholtz_1d",
    "laplace",
    "stokes",
}

_FACTORY_EXPORTS = {"Kernel", "kernel"}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _FACTORY_EXPORTS:
        value = getattr(import_module(f"{__name__}.factory"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "biharmonic",
    "elasticity",
    "factory",
    "helmholtz",
    "helmholtz_1d",
    "Kernel",
    "kernel",
    "laplace",
    "stokes",
]
