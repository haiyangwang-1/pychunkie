"""Quadrature and corner-compression routines.

This package is the Python-facing home for native smooth assembly, GGQ/adaptive
special quadrature, panel product quadrature, and RCIP helpers. The old
``chunkie.chnk`` module names remain as MATLAB-compatible wrappers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SUBMODULES = {"adaptive", "ggq", "native", "panel", "rcip"}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "adaptive",
    "ggq",
    "native",
    "panel",
    "rcip",
]
