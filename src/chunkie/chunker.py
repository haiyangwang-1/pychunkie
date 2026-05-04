"""Chunked curve data structure mirroring MATLAB ``@chunker``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import lege


@dataclass
class ChunkerPref:
    """Preferences for constructing a :class:`Chunker`."""

    nchmax: int = 10000
    k: int = 16
    dim: int = 2
    nchstor: int = 4
    verttol: float = 1e-12

    @classmethod
    def from_any(cls, pref: "ChunkerPref | dict[str, Any] | None" = None) -> "ChunkerPref":
        if pref is None:
            return cls()
        if isinstance(pref, cls):
            return pref
        if isinstance(pref, dict):
            values = cls().__dict__.copy()
            values.update(pref)
            return cls(**values)
        raise TypeError("pref must be None, a dict, or ChunkerPref")


class Chunker:
    """Curve divided into Legendre-discretized chunks.

    Arrays follow the MATLAB layout ``dim x k x nch``. The class starts with
    zero chunks; call :meth:`addchunk` and then fill ``r``, ``d``, and ``d2``.
    """

    __array_priority__ = 1000

    def __init__(
        self,
        pref: ChunkerPref | dict[str, Any] | None = None,
        t: ArrayLike | None = None,
        w: ArrayLike | None = None,
    ) -> None:
        p = ChunkerPref.from_any(pref)
        if p.k < 2:
            raise ValueError("CHUNKER: order k of panels must be at least 2")
        if p.nchstor < 0 or p.nchmax < 0:
            raise ValueError("chunk storage sizes must be non-negative")
        if p.nchstor > p.nchmax:
            raise ValueError("nchstor must not exceed nchmax")

        if t is None or w is None:
            self.tstor, self.wstor, _, _ = lege.exps(p.k)
        else:
            self.tstor = np.asarray(t, dtype=float)
            self.wstor = np.asarray(w, dtype=float)
            if self.tstor.size != p.k or self.wstor.size != p.k:
                raise ValueError("precomputed Legendre nodes appear to be wrong order")

        self.verttol = p.verttol
        self.nchmax = int(p.nchmax)
        self.nchstor = int(p.nchstor)
        self.nch = 0
        self.rstor = np.zeros((p.dim, p.k, self.nchstor))
        self.dstor = np.zeros_like(self.rstor)
        self.d2stor = np.zeros_like(self.rstor)
        self.nstor = np.zeros_like(self.rstor)
        self.wtsstor = np.zeros((p.k, self.nchstor))
        self.adjstor = np.zeros((2, self.nchstor), dtype=int)
        self.datastor = np.zeros((0, p.k, self.nchstor))
        self.hasdata = False
        self.vert: list[np.ndarray] = []

    @property
    def k(self) -> int:
        return self.rstor.shape[1]

    @property
    def dim(self) -> int:
        return self.rstor.shape[0]

    @property
    def npt(self) -> int:
        return self.k * self.nch

    @property
    def datadim(self) -> int:
        return self.datastor.shape[0] if self.hasdata else 0

    @property
    def nvert(self) -> int:
        return len(self.vert)

    @property
    def vertdeg(self) -> np.ndarray:
        return np.array([len(v) for v in self.vert], dtype=int)

    @property
    def r(self) -> np.ndarray:
        return self.rstor[:, :, : self.nch]

    @r.setter
    def r(self, value: ArrayLike) -> None:
        self.rstor[:, :, : self.nch] = value

    @property
    def d(self) -> np.ndarray:
        return self.dstor[:, :, : self.nch]

    @d.setter
    def d(self, value: ArrayLike) -> None:
        self.dstor[:, :, : self.nch] = value

    @property
    def d2(self) -> np.ndarray:
        return self.d2stor[:, :, : self.nch]

    @d2.setter
    def d2(self, value: ArrayLike) -> None:
        self.d2stor[:, :, : self.nch] = value

    @property
    def n(self) -> np.ndarray:
        return self.nstor[:, :, : self.nch]

    @n.setter
    def n(self, value: ArrayLike) -> None:
        self.nstor[:, :, : self.nch] = value

    @property
    def wts(self) -> np.ndarray:
        return self.wtsstor[:, : self.nch]

    @wts.setter
    def wts(self, value: ArrayLike) -> None:
        self.wtsstor[:, : self.nch] = value

    @property
    def adj(self) -> np.ndarray:
        return self.adjstor[:, : self.nch]

    @adj.setter
    def adj(self, value: ArrayLike) -> None:
        self.adjstor[:, : self.nch] = value

    @property
    def data(self) -> np.ndarray:
        if not self.hasdata:
            return np.zeros((0, self.k, 0))
        return self.datastor[:, :, : self.nch]

    @data.setter
    def data(self, value: ArrayLike) -> None:
        if not self.hasdata:
            raise ValueError("data rows have not been allocated")
        self.datastor[:, :, : self.nch] = value

    def copy(self) -> "Chunker":
        other = Chunker(
            ChunkerPref(
                nchmax=self.nchmax,
                k=self.k,
                dim=self.dim,
                nchstor=self.nchstor,
                verttol=self.verttol,
            ),
            self.tstor.copy(),
            self.wstor.copy(),
        )
        other.nch = self.nch
        other.rstor = self.rstor.copy()
        other.dstor = self.dstor.copy()
        other.d2stor = self.d2stor.copy()
        other.nstor = self.nstor.copy()
        other.wtsstor = self.wtsstor.copy()
        other.adjstor = self.adjstor.copy()
        other.datastor = self.datastor.copy()
        other.hasdata = self.hasdata
        other.vert = [v.copy() for v in self.vert]
        return other

    def addchunk(self, nchadd: int = 1) -> "Chunker":
        if int(nchadd) != nchadd or nchadd <= 0:
            raise ValueError("nchadd must be positive integer")
        nchadd = int(nchadd)
        if self.nch + nchadd > self.nchmax:
            raise ValueError("adding chunks would exceed maximum chunker length")
        while self.nch + nchadd > self.nchstor:
            target = max(
                min(2 * max(self.nchstor, 1), self.nchmax),
                min(self.nch + nchadd, self.nchmax),
            )
            self.resize(target)
        self.nch += nchadd
        return self

    def resize(self, nchstornew: int) -> "Chunker":
        if nchstornew < self.nch:
            raise ValueError("new storage is less than number of chunks")
        if nchstornew > self.nchmax:
            raise ValueError("new storage exceeds maximum storage")

        def grow(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            out = np.zeros(shape, dtype=arr.dtype)
            slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
            out[slices] = arr[slices]
            return out

        self.rstor = grow(self.rstor, (self.dim, self.k, nchstornew))
        self.dstor = grow(self.dstor, (self.dim, self.k, nchstornew))
        self.d2stor = grow(self.d2stor, (self.dim, self.k, nchstornew))
        self.nstor = grow(self.nstor, (self.dim, self.k, nchstornew))
        self.wtsstor = grow(self.wtsstor, (self.k, nchstornew))
        self.adjstor = grow(self.adjstor, (2, nchstornew))
        self.datastor = grow(self.datastor, (self.datastor.shape[0], self.k, nchstornew))
        self.nchstor = int(nchstornew)
        return self

    def makedatarows(self, nrows: int) -> "Chunker":
        if nrows <= 0:
            return self
        old = self.datastor
        self.datastor = np.zeros((old.shape[0] + int(nrows), self.k, self.nchstor))
        self.datastor[: old.shape[0], :, : old.shape[2]] = old
        self.hasdata = True
        return self

    def cleardata(self) -> "Chunker":
        self.hasdata = False
        self.datastor = np.zeros((0, self.k, self.nchstor))
        return self

    def weights(self) -> np.ndarray:
        speed = np.sqrt(np.sum(np.abs(self.d) ** 2, axis=0))
        return speed * self.wstor[:, None]

    def normals(self) -> np.ndarray:
        if self.dim != 2:
            raise ValueError("normals only implemented for dim=2")
        speed = np.sqrt(self.dstor[0, :, : self.nch] ** 2 + self.dstor[1, :, : self.nch] ** 2)
        out = np.zeros((2, self.k, self.nch), dtype=self.dstor.dtype)
        out[0] = self.dstor[1, :, : self.nch] / speed
        out[1] = -self.dstor[0, :, : self.nch] / speed
        return out

    def tangents(self) -> np.ndarray:
        speed = np.sqrt(np.sum(np.abs(self.d) ** 2, axis=0))
        return self.d / speed[None, :, :]

    def chunklen(self, ich: ArrayLike | None = None) -> np.ndarray:
        if ich is None:
            return np.sum(self.wts, axis=0)
        indices = np.asarray(ich, dtype=int)
        return np.sum(self.wts[:, indices], axis=0)

    def area(self) -> float:
        if self.dim != 2:
            raise ValueError("area only well-defined for 2d chunkers")
        if np.any(self.adj == 0):
            raise ValueError("area only well-defined for closed 2d chunkers")
        if np.any(self.vertdeg > 2):
            raise ValueError("area not well-defined for higher order vertices")
        integrand = np.sum(self.n * self.r, axis=0)
        return float(np.sum(self.wts * integrand) / self.dim)

    def min(self) -> np.ndarray:
        if self.nch == 0:
            return np.full(self.dim, np.nan)
        return np.min(np.real(self.r.reshape(self.dim, self.npt)), axis=1)

    def max(self) -> np.ndarray:
        if self.nch == 0:
            return np.full(self.dim, np.nan)
        return np.max(np.real(self.r.reshape(self.dim, self.npt)), axis=1)

    def recompute_geometry(self) -> "Chunker":
        self.n = self.normals()
        self.wts = self.weights()
        return self

    def translate(self, vector: ArrayLike) -> "Chunker":
        vec = np.asarray(vector, dtype=self.rstor.dtype).reshape(-1)
        if vec.size != self.dim:
            raise ValueError("translation vector has incompatible dimension")
        out = self.copy()
        out.r = out.r + vec[:, None, None]
        return out

    def transform(self, matrix: ArrayLike) -> "Chunker":
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

    def __add__(self, other: ArrayLike) -> "Chunker":
        return self.translate(other)

    def __radd__(self, other: ArrayLike) -> "Chunker":
        return self.translate(other)

    def __mul__(self, other: Any) -> "Chunker":
        if np.isscalar(other):
            return self.transform(other)
        raise TypeError("product of chunker and matrix only defined for matrix on left")

    def __rmul__(self, other: Any) -> "Chunker":
        return self.transform(other)

    def __rmatmul__(self, other: Any) -> "Chunker":
        return self.transform(other)


def chunker(
    pref: ChunkerPref | dict[str, Any] | None = None,
    t: ArrayLike | None = None,
    w: ArrayLike | None = None,
) -> Chunker:
    """MATLAB-style constructor alias."""

    return Chunker(pref, t, w)


def chunkerpref(pref: ChunkerPref | dict[str, Any] | None = None) -> ChunkerPref:
    """MATLAB-style preference constructor alias."""

    return ChunkerPref.from_any(pref)
