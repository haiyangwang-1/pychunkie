"""Private chunkertransformmixin methods for :class:`chunkie.geometry.Chunker`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ._chunker_class import Chunker


class ChunkerTransformMixin:
    def translate(self, vector: ArrayLike) -> Chunker:
        vec = np.asarray(vector, dtype=self.rstor.dtype).reshape(-1)
        if vec.size != self.dim:
            raise ValueError("translation vector has incompatible dimension")
        out = self.copy()
        out.r = out.r + vec[:, None, None]
        return out

    def transform(self, matrix: ArrayLike) -> Chunker:
        mat = np.asarray(matrix, dtype=self.rstor.dtype)
        if mat.ndim == 0:
            mat = mat * np.eye(self.dim)
        if mat.shape != (self.dim, self.dim):
            raise ValueError("matrix must have compatible size for transforming coordinates")
        out = self.copy()
        out.r = np.einsum("ij,jkl->ikl", mat, out.r)
        out.d = np.einsum("ij,jkl->ikl", mat, out.d)
        out.d2 = np.einsum("ij,jkl->ikl", mat, out.d2)
        out.recompute_geometry()
        return out

    def reverse(self) -> Chunker:
        out = self.copy()
        out.r = out.r[:, ::-1, :]
        out.d = -out.d[:, ::-1, :]
        out.d2 = out.d2[:, ::-1, :]
        out.adj = out.adj[::-1, :]
        out.n = -out.n[:, ::-1, :]
        out.wts = out.wts[::-1, :]
        if out.hasdata:
            out.data = out.data[:, ::-1, :]
        return out

    def move(
        self,
        r0: ArrayLike | None = None,
        r1: ArrayLike | None = None,
        trotat: float = 0.0,
        scale: float = 1.0,
    ) -> Chunker:
        if self.dim != 2:
            raise ValueError("move is implemented for 2D chunkers")
        center0 = np.zeros(2) if r0 is None else np.asarray(r0, dtype=float).reshape(2)
        center1 = np.zeros(2) if r1 is None else np.asarray(r1, dtype=float).reshape(2)
        rot = np.array([[np.cos(trotat), -np.sin(trotat)], [np.sin(trotat), np.cos(trotat)]])
        out = self.copy()
        out.r = (
            scale * np.einsum("ij,jkl->ikl", rot, out.r - center0[:, None, None])
            + center1[:, None, None]
        )
        out.d = scale * np.einsum("ij,jkl->ikl", rot, out.d)
        out.d2 = scale * np.einsum("ij,jkl->ikl", rot, out.d2)
        normal_sign = -1.0 if scale < 0 else 1.0
        out.n = normal_sign * np.einsum("ij,jkl->ikl", rot, out.n)
        out.wts = out.weights()
        return out

    def rotate(
        self,
        theta: float = 0.0,
        r0: ArrayLike | None = None,
        r1: ArrayLike | None = None,
    ) -> Chunker:
        if self.dim != 2:
            raise ValueError("rotate is implemented for 2D chunkers")
        if not np.isreal(theta):
            raise ValueError("rotate only supports real angles")
        center0 = np.zeros(2) if r0 is None else np.asarray(r0, dtype=float).reshape(2)
        center1 = np.zeros(2) if r1 is None else np.asarray(r1, dtype=float).reshape(2)
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        rot = np.array([[c, -s], [s, c]])
        out = self.copy()
        out.r = (
            np.einsum("ij,jkl->ikl", rot, out.r - center0[:, None, None]) + center1[:, None, None]
        )
        out.d = np.einsum("ij,jkl->ikl", rot, out.d)
        out.d2 = np.einsum("ij,jkl->ikl", rot, out.d2)
        out.n = np.einsum("ij,jkl->ikl", rot, out.n)
        return out

    def reflect(
        self,
        theta: float = 0.0,
        r0: ArrayLike | None = None,
        r1: ArrayLike | None = None,
    ) -> Chunker:
        if self.dim != 2:
            raise ValueError("reflect is implemented for 2D chunkers")
        if not np.isreal(theta):
            raise ValueError("reflect only supports real angles")
        center0 = np.zeros(2) if r0 is None else np.asarray(r0, dtype=float).reshape(2)
        center1 = np.zeros(2) if r1 is None else np.asarray(r1, dtype=float).reshape(2)
        c = float(np.cos(2.0 * theta))
        s = float(np.sin(2.0 * theta))
        refmat = np.array([[c, s], [s, -c]])
        out = self.copy()
        out.r = (
            np.einsum("ij,jkl->ikl", refmat, out.r - center0[:, None, None])
            + center1[:, None, None]
        )
        out.d = np.einsum("ij,jkl->ikl", refmat, out.d)
        out.d2 = np.einsum("ij,jkl->ikl", refmat, out.d2)
        out.n = np.einsum("ij,jkl->ikl", refmat, out.n)
        return out

    def __add__(self, other: ArrayLike) -> Chunker:
        return self.translate(other)

    def __radd__(self, other: ArrayLike) -> Chunker:
        return self.translate(other)

    def __mul__(self, other: Any) -> Chunker:
        if np.isscalar(other):
            return self.transform(other)
        raise TypeError("product of chunker and matrix only defined for matrix on left")

    def __rmul__(self, other: Any) -> Chunker:
        return self.transform(other)

    def __rmatmul__(self, other: Any) -> Chunker:
        return self.transform(other)


_LEGACY_OPTIONS_MARKER = "_chunkie_normalized_geometry_options"
