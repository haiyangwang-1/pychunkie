"""Utilities mirroring MATLAB ``+chnk``.

Submodules are imported on first access so lightweight imports such as
``chunkie.kernel`` do not eagerly load every geometry, FLAM, and kernel helper.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SUBMODULES = {
    "biharm2d",
    "elast2d",
    "flam",
    "helm1d",
    "helm2d",
    "lap2d",
    "stok2d",
}

def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "biharm2d",
    "elast2d",
    "flam",
    "helm1d",
    "helm2d",
    "lap2d",
    "stok2d",
]
