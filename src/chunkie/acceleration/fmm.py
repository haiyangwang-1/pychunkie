"""Optional FMM2D backend loader."""

from __future__ import annotations

from typing import Any


def load_fmm2dpy() -> Any | None:
    """Return the optional ``fmm2dpy`` module, or ``None`` when unavailable."""

    try:  # pragma: no cover - exercised when the optional compiled package imports.
        import fmm2dpy
    except (ImportError, OSError):  # pragma: no cover - keep source installs usable without FMM2D.
        return None
    return fmm2dpy


fmm2dpy = load_fmm2dpy()

__all__ = ["fmm2dpy", "load_fmm2dpy"]
