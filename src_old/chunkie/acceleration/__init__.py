"""Acceleration helper modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SUBMODULES = {"flam", "fmm", "fmm_kernel"}
_EXPORTS = {"FmmKernel"}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _EXPORTS:
        value = getattr(import_module(f"{__name__}.fmm_kernel"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FmmKernel", "flam", "fmm", "fmm_kernel"]
