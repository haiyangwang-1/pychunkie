"""Numerical helper modules used by geometry and quadrature code."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SUBMODULES = {"arcparam", "smoother", "special"}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["arcparam", "smoother", "special"]
