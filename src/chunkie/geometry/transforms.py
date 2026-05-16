"""Geometry transform helpers."""

from __future__ import annotations

from numpy.typing import ArrayLike

from .chunker import Chunker


def translate(chunker: Chunker, vector: ArrayLike) -> Chunker:
    return chunker.translated(vector)


def scale(chunker: Chunker, factor: float, *, center: ArrayLike | None = None) -> Chunker:
    return chunker.scaled(factor, center=center)
