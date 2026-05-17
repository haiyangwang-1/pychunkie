"""Geometry transform helpers."""

from __future__ import annotations

from numpy.typing import ArrayLike

from .chunker import Chunker


def translate(chunker: Chunker, vector: ArrayLike) -> Chunker:
    return chunker.translated(vector)


def scale(chunker: Chunker, factor: float, *, center: ArrayLike | None = None) -> Chunker:
    return chunker.scaled(factor, center=center)


def affine(chunker: Chunker, matrix: ArrayLike, *, offset: ArrayLike | None = None) -> Chunker:
    return chunker.affine(matrix, offset=offset)


def rotate(
    chunker: Chunker,
    angle: float,
    *,
    center: ArrayLike | None = None,
    target_center: ArrayLike | None = None,
) -> Chunker:
    return chunker.rotated(angle, center=center, target_center=target_center)


def reflect(
    chunker: Chunker,
    angle: float,
    *,
    center: ArrayLike | None = None,
    target_center: ArrayLike | None = None,
) -> Chunker:
    return chunker.reflected(angle, center=center, target_center=target_center)
