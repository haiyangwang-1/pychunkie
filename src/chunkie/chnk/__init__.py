"""Utilities mirroring MATLAB ``+chnk``.

Submodules are imported on first access so lightweight imports such as
``chunkie.kernel`` do not eagerly load every quadrature, FLAM, and RCIP helper.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SUBMODULES = {
    "arcparam",
    "biharm2d",
    "curves",
    "elast2d",
    "flam",
    "geometry",
    "helm1d",
    "helm2d",
    "lap2d",
    "pquad",
    "quadadap",
    "quadggq",
    "quadnative",
    "rcip",
    "smoother",
    "spcl",
    "stok2d",
}

_GEOMETRY_EXPORTS = {
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
    if name in _GEOMETRY_EXPORTS:
        value = getattr(import_module(f"{__name__}.geometry"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "chunk_nearparam",
    "arcparam",
    "biharm2d",
    "curvature2d",
    "curves",
    "elast2d",
    "flam",
    "flagnear",
    "flagnear_rectangle",
    "flagnear_rectangle_grid",
    "flagself",
    "geometry",
    "helm1d",
    "helm2d",
    "lap2d",
    "normal2d",
    "perp",
    "pquad",
    "quadadap",
    "quadggq",
    "quadnative",
    "rcip",
    "smoother",
    "spcl",
    "stok2d",
]
