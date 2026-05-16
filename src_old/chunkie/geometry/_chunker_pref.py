"""Chunker construction preferences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ChunkerPref:
    """Preferences for constructing a :class:`Chunker`.

    ``k`` is the Gauss-Legendre order per chunk, ``dim`` is the ambient
    coordinate dimension, and ``nchmax``/``nchstor`` control maximum and initial
    chunk storage. ``verttol`` is used by topology helpers that compare chunk
    endpoints.
    """

    nchmax: int = 10000
    k: int = 16
    dim: int = 2
    nchstor: int = 4
    verttol: float = 1e-12

    @classmethod
    def from_any(cls, pref: ChunkerPref | dict[str, Any] | None = None) -> ChunkerPref:
        if pref is None:
            return cls()
        if isinstance(pref, cls):
            return pref
        if isinstance(pref, dict):
            values = cls().__dict__.copy()
            values.update(pref)
            return cls(**values)
        raise TypeError("pref must be None, a dict, or ChunkerPref")
