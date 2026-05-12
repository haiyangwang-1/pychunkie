"""Geometry helper modules and point/panel predicates."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SUBMODULES = {"curves", "predicates"}

_PREDICATE_EXPORTS = {
    "chunk_nearparam",
    "curvature2d",
    "flagnear",
    "flagnear_rectangle",
    "flagnear_rectangle_grid",
    "flagself",
    "normal2d",
    "perp",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _PREDICATE_EXPORTS:
        value = getattr(import_module(f"{__name__}.predicates"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "chunk_nearparam",
    "curvature2d",
    "curves",
    "flagnear",
    "flagnear_rectangle",
    "flagnear_rectangle_grid",
    "flagself",
    "normal2d",
    "perp",
    "predicates",
]
