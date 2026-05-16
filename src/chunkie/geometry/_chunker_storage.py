"""Private chunkerstoragemixin methods for :class:`chunkie.geometry.Chunker`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .. import lege
from ._chunker_pref import ChunkerPref

if TYPE_CHECKING:
    from ._chunker_class import Chunker


class ChunkerStorageMixin:
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
    def quadrature_order(self) -> int:
        return self.k

    @property
    def dim(self) -> int:
        return self.rstor.shape[0]

    @property
    def coordinate_dim(self) -> int:
        return self.dim

    @property
    def npt(self) -> int:
        return self.k * self.nch

    @property
    def point_count(self) -> int:
        return self.npt

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
    def positions(self) -> np.ndarray:
        return self.r

    @positions.setter
    def positions(self, value: ArrayLike) -> None:
        self.r = value

    @property
    def d(self) -> np.ndarray:
        return self.dstor[:, :, : self.nch]

    @d.setter
    def d(self, value: ArrayLike) -> None:
        self.dstor[:, :, : self.nch] = value

    @property
    def derivatives(self) -> np.ndarray:
        return self.d

    @derivatives.setter
    def derivatives(self, value: ArrayLike) -> None:
        self.d = value

    @property
    def d2(self) -> np.ndarray:
        return self.d2stor[:, :, : self.nch]

    @d2.setter
    def d2(self, value: ArrayLike) -> None:
        self.d2stor[:, :, : self.nch] = value

    @property
    def second_derivatives(self) -> np.ndarray:
        return self.d2

    @second_derivatives.setter
    def second_derivatives(self, value: ArrayLike) -> None:
        self.d2 = value

    @property
    def n(self) -> np.ndarray:
        return self.nstor[:, :, : self.nch]

    @n.setter
    def n(self, value: ArrayLike) -> None:
        self.nstor[:, :, : self.nch] = value

    @property
    def normal_vectors(self) -> np.ndarray:
        return self.n

    @normal_vectors.setter
    def normal_vectors(self, value: ArrayLike) -> None:
        self.n = value

    @property
    def wts(self) -> np.ndarray:
        return self.wtsstor[:, : self.nch]

    @wts.setter
    def wts(self, value: ArrayLike) -> None:
        self.wtsstor[:, : self.nch] = value

    @property
    def quadrature_weights(self) -> np.ndarray:
        return self.wts

    @quadrature_weights.setter
    def quadrature_weights(self, value: ArrayLike) -> None:
        self.wts = value

    @property
    def adj(self) -> np.ndarray:
        return self.adjstor[:, : self.nch]

    @adj.setter
    def adj(self, value: ArrayLike) -> None:
        self.adjstor[:, : self.nch] = value

    @property
    def adjacency(self) -> np.ndarray:
        return self.adj

    @adjacency.setter
    def adjacency(self, value: ArrayLike) -> None:
        self.adj = value

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

    def copy(self) -> Chunker:
        other = type(self)(
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

    def addchunk(self, nchadd: int = 1) -> Chunker:
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

    def resize(self, nchstornew: int) -> Chunker:
        if nchstornew < self.nch:
            raise ValueError("new storage is less than number of chunks")
        if nchstornew > self.nchmax:
            raise ValueError("new storage exceeds maximum storage")

        def grow(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            out = np.zeros(shape, dtype=arr.dtype)
            slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape, strict=False))
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

    def makedatarows(self, nrows: int) -> Chunker:
        if nrows <= 0:
            return self
        old = self.datastor
        self.datastor = np.zeros((old.shape[0] + int(nrows), self.k, self.nchstor))
        self.datastor[: old.shape[0], :, : old.shape[2]] = old
        self.hasdata = True
        return self

    def cleardata(self) -> Chunker:
        self.hasdata = False
        self.datastor = np.zeros((0, self.k, self.nchstor))
        return self
