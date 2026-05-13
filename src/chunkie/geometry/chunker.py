"""Chunked curve data structure mirroring MATLAB ``@chunker``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike

from .. import lege
from ._chunker_polygon import (
    _dyadic_chunkerpoly,
    _fill_line_chunk,
    _fill_quadratic_chunk,
    _polygon_widths,
    _rounded_chunkerpoly,
)
from ._nearest import chunk_nearparam as _chunk_nearparam


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

    Arrays follow the MATLAB layout ``dim x k x nch``:

    - ``r`` stores node positions.
    - ``d`` and ``d2`` store first and second derivatives with respect to the
      local panel parameter.
    - ``n`` and ``wts`` store outward normals and physical quadrature weights.
    - ``adj`` stores one-based MATLAB-style neighboring chunk labels, with
      nonpositive entries denoting open ends.

    User code usually calls :func:`chunkerfunc`, :func:`chunkerpoly`,
    :func:`chunkerfit`, or :func:`chunkerpoints` instead of filling this storage
    manually. Matrix assembly flattens nodes in Fortran order, matching MATLAB
    chunk-contiguous ordering.
    """

    __array_priority__ = 1000
    lvlrfacdefault = 2.1

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
        """Return physical Gauss weights ``|dr/dt| * w`` for every node."""

        speed = np.sqrt(np.sum(np.abs(self.d) ** 2, axis=0))
        return speed * self.wstor[:, None]

    def normals(self) -> np.ndarray:
        """Return outward 2D normals from the stored panel derivatives."""

        if self.dim != 2:
            raise ValueError("normals only implemented for dim=2")
        speed = np.sqrt(self.dstor[0, :, : self.nch] ** 2 + self.dstor[1, :, : self.nch] ** 2)
        out = np.zeros((2, self.k, self.nch), dtype=self.dstor.dtype)
        out[0] = self.dstor[1, :, : self.nch] / speed
        out[1] = -self.dstor[0, :, : self.nch] / speed
        return out

    def tangents(self) -> np.ndarray:
        """Return unit tangent vectors at all panel nodes."""

        speed = np.sqrt(np.sum(np.abs(self.d) ** 2, axis=0))
        return self.d / speed[None, :, :]

    def arclengthdens(self) -> np.ndarray:
        return np.sqrt(np.sum(self.d**2, axis=0))

    def arclengthder(self, u: ArrayLike) -> np.ndarray:
        dmat = lege.dermat(self.k)
        vals = np.asarray(u).reshape(self.k, self.nch)
        return (dmat @ vals) / self.arclengthdens()

    def arclengthfun(self) -> np.ndarray:
        aint = lege.intmat(self.k)[0]
        s = aint @ self.arclengthdens()
        starts = np.zeros(self.nch)
        inds, _, info = self.sortinfo()
        chunklens = self.chunklen()
        offset = 0
        for nch in np.asarray(info["nchs"], dtype=int):
            sstart = 0.0
            for idx in inds[offset : offset + nch]:
                starts[idx] = sstart
                sstart += chunklens[idx]
            offset += nch
        return s + starts[None, :]

    def chunklen(self, ich: ArrayLike | None = None) -> np.ndarray:
        """Return arclengths for selected chunks or for all chunks."""

        if ich is None:
            return np.sum(self.wts, axis=0)
        indices = np.asarray(ich, dtype=int)
        return np.sum(self.wts[:, indices], axis=0)

    def chunkends(self, ich: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
        if ich is None:
            indices = np.arange(self.nch)
        else:
            indices = np.asarray(ich, dtype=int)
        pends = lege.matrin(self.k, np.array([-1.0, 1.0]))[0]
        rend = np.zeros((self.dim, 2, indices.size), dtype=self.rstor.dtype)
        tauend = np.zeros_like(rend)
        for j, idx in enumerate(indices):
            rend[:, :, j] = (pends @ self.r[:, :, idx].T).T
            dend = (pends @ self.d[:, :, idx].T).T
            tauend[:, :, j] = dend / np.sqrt(np.sum(np.abs(dend) ** 2, axis=0))[None, :]
        return rend, tauend

    def area(self) -> float:
        """Return the signed area enclosed by a closed 2D chunker."""

        if self.dim != 2:
            raise ValueError("area only well-defined for 2d chunkers")
        if np.any(self.adj == 0):
            raise ValueError("area only well-defined for closed 2d chunkers")
        if np.any(self.vertdeg > 2):
            raise ValueError("area not well-defined for higher order vertices")
        integrand = np.sum(self.n * self.r, axis=0)
        return float(np.sum(self.wts * integrand) / self.dim)

    def signed_curvature(self) -> np.ndarray:
        if self.dim != 2:
            raise ValueError("signed curvature only defined in 2D")
        speed = np.sqrt(np.sum(self.d**2, axis=0))
        return (self.d[0] * self.d2[1] - self.d[1] * self.d2[0]) / speed**3

    def exps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, _, u, _ = lege.exps(self.k)
        rc = np.einsum("ij,djn->din", u, self.r)
        dc = np.einsum("ij,djn->din", u, self.d)
        d2c = np.einsum("ij,djn->din", u, self.d2)
        return rc, dc, d2c

    def diffmat(self, order: int = 1) -> np.ndarray:
        if int(order) != order or order < 0:
            raise ValueError("Differentiation order must be a nonnegative integer")
        dleg = lege.dermat(self.k)
        out = np.zeros((self.npt, self.npt))
        for ich in range(self.nch):
            block = dleg / self.arclengthdens()[:, ich][:, None]
            block_power = np.linalg.matrix_power(block, int(order))
            idx = slice(ich * self.k, (ich + 1) * self.k)
            out[idx, idx] = block_power
        return out

    def intmat(self) -> np.ndarray:
        """Return the arc-length integration matrix along the chunk order."""

        panel_int = lege.intmat(self.k)[0]
        ds = self.arclengthdens()
        out = np.zeros((self.npt, self.npt), dtype=np.result_type(self.rstor, float))

        order: list[int] = []
        seen: set[int] = set()
        cur = 0
        while 0 <= cur < self.nch and cur not in seen:
            order.append(cur)
            seen.add(cur)
            nxt = int(self.adj[1, cur])
            if nxt <= 0:
                break
            cur = nxt - 1
        order.extend(idx for idx in range(self.nch) if idx not in seen)

        for pos, ich in enumerate(order):
            rows = slice(ich * self.k, (ich + 1) * self.k)
            out[rows, rows] = panel_int * ds[:, ich][None, :]
            for prev in order[:pos]:
                cols = slice(prev * self.k, (prev + 1) * self.k)
                out[rows, cols] = np.ones((self.k, 1)) @ self.wts[:, prev][None, :]
        return out

    def onesmat(self) -> np.ndarray:
        wts = self.wts.reshape(-1, order="F")
        return np.ones((self.npt, 1)) @ wts[None, :]

    def normonesmat(self) -> np.ndarray:
        normals = self.n.reshape(-1, order="F")
        wts2 = (np.repeat(self.wts.reshape(-1, order="F"), self.dim) * normals)
        return normals[:, None] @ wts2[None, :]

    def centroids(self) -> np.ndarray:
        return np.sum(self.r * self.wstor[None, :, None], axis=1) / 2.0

    def datares(self, opts: dict[str, Any] | None = None) -> np.ndarray:
        """Check whether selected data rows are Legendre-resolved per chunk."""

        opts = {} if opts is None else dict(opts)
        if not self.hasdata or self.datadim == 0:
            return np.zeros((0, self.nch), dtype=bool)

        idata = np.asarray(opts.get("idata", np.arange(self.datadim)), dtype=int).reshape(-1)
        if np.any(idata < 0) or np.any(idata >= self.datadim):
            raise IndexError("data row index out of range")

        ncoeff = int(opts.get("ncoeff", np.floor((self.k + 0.1) / 2.0)))
        ncoeff = min(max(ncoeff, 1), self.k)
        pleg = opts.get("pleg", 1)
        tol = float(opts.get("tol", 1.0e-6))
        pscale = float(opts.get("pscale", 0.0))
        rel = bool(opts.get("rel", False))

        _, _, u, _ = lege.exps(self.k)
        tail = u[self.k - ncoeff :, :]
        head = u[: self.k - ncoeff, :]
        flags = np.zeros((idata.size, self.nch), dtype=bool)
        lens = self.chunklen()

        for ich in range(self.nch):
            datai = self.data[idata, :, ich]
            tail_coeffs = tail @ datai.T
            tail_norm = np.linalg.norm(tail_coeffs, ord=pleg, axis=0)
            scaled_tail = tail_norm * lens[ich] ** pscale
            if rel:
                if head.shape[0] == 0:
                    head_norm = np.zeros_like(tail_norm)
                else:
                    head_norm = np.linalg.norm(head @ datai.T, ord=pleg, axis=0)
                flags[:, ich] = scaled_tail < tol * head_norm
            else:
                flags[:, ich] = scaled_tail < tol
        return flags

    def sortinfo(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        ier = 0
        for idx in range(self.nch):
            left, right = self.adj[:, idx]
            if right > 0 and self.adj[0, right - 1] != idx + 1:
                ier = 1
            if left > 0 and self.adj[1, left - 1] != idx + 1:
                ier = 1

        visited = np.zeros(self.nch, dtype=bool)
        order: list[int] = []
        nchs: list[int] = []
        ifclosed: list[bool] = []

        starts = [i for i in range(self.nch) if self.adj[0, i] <= 0]
        for start in starts:
            if visited[start]:
                continue
            comp: list[int] = []
            cur = start
            closed = False
            while 0 <= cur < self.nch and not visited[cur]:
                comp.append(cur)
                visited[cur] = True
                nxt = int(self.adj[1, cur])
                if nxt <= 0:
                    break
                cur = nxt - 1
                if cur == start:
                    closed = True
                    break
            order.extend(comp)
            nchs.append(len(comp))
            ifclosed.append(closed)

        for start in range(self.nch):
            if visited[start]:
                continue
            comp = []
            cur = start
            closed = True
            while not visited[cur]:
                comp.append(cur)
                visited[cur] = True
                nxt = int(self.adj[1, cur])
                if nxt <= 0:
                    closed = False
                    break
                cur = nxt - 1
                if cur < 0 or cur >= self.nch:
                    ier = 1
                    closed = False
                    break
            order.extend(comp)
            nchs.append(len(comp))
            ifclosed.append(closed)

        if len(order) != self.nch or len(set(order)) != self.nch:
            ier = 2
            order = list(range(self.nch))

        inds = np.array(order, dtype=int)
        adjs = _remap_adjacency(self.adj[:, inds], inds)
        info = {
            "ncomp": len(nchs),
            "nchs": np.array(nchs, dtype=int),
            "ifclosed": np.array(ifclosed, dtype=bool),
            "ier": ier,
        }
        return inds, adjs, info

    def checkadjinfo(self) -> int:
        return int(self.sortinfo()[2]["ier"])

    def sort(self) -> tuple["Chunker", dict[str, Any]]:
        inds, adjs, info = self.sortinfo()
        out = self.copy()
        out.r = out.r[:, :, inds]
        out.d = out.d[:, :, inds]
        out.d2 = out.d2[:, :, inds]
        out.n = out.n[:, :, inds]
        out.wts = out.wts[:, inds]
        if out.hasdata:
            out.data = out.data[:, :, inds]
        out.adj = adjs
        return out, info

    def flagnear(self, pts: ArrayLike, opts: dict[str, Any] | None = None, *, fac: float | None = None) -> np.ndarray:
        opts = _legacy_options(opts, "flagnear opts")
        _set_option(opts, "fac", fac)
        fac = float(opts.get("fac", 1.0))
        points = np.asarray(pts, dtype=float).reshape(self.dim, -1)
        flags = np.zeros((points.shape[1], self.nch), dtype=bool)
        lens = self.chunklen() * fac
        for ich in range(self.nch):
            diff = points[:, :, None] - self.r[:, :, ich][:, None, :]
            dists = np.sqrt(np.sum(diff**2, axis=0))
            flags[:, ich] = np.any(dists < lens[ich], axis=1)
        return flags

    def flagnear_rectangle(
        self,
        pts: ArrayLike,
        opts: dict[str, Any] | None = None,
        *,
        rho: float | None = None,
    ) -> np.ndarray:
        opts = _legacy_options(opts, "flagnear_rectangle opts")
        _set_option(opts, "rho", rho)
        if self.dim != 2:
            raise ValueError("flagnear_rectangle is implemented for 2D chunkers")
        rho = float(opts.get("rho", 1.8))
        rectinfo = _bernstein_rectangle_info(self, rho)
        points = np.asarray(pts, dtype=float).reshape(2, -1)
        flags = np.zeros((points.shape[1], self.nch), dtype=bool)
        for ich in range(self.nch):
            d1 = points.T @ rectinfo[:, 0, ich]
            d2 = points.T @ rectinfo[:, 1, ich]
            flags[:, ich] = (
                (d1 >= rectinfo[0, 2, ich])
                & (d1 <= rectinfo[1, 2, ich])
                & (d2 >= rectinfo[0, 3, ich])
                & (d2 <= rectinfo[1, 3, ich])
            )
        return flags

    def flagnear_rectangle_grid(
        self,
        x: ArrayLike,
        y: ArrayLike,
        opts: dict[str, Any] | None = None,
        *,
        rho: float | None = None,
    ) -> np.ndarray:
        opts = _legacy_options(opts, "flagnear_rectangle_grid opts")
        _set_option(opts, "rho", rho)
        xx, yy = np.meshgrid(np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1))
        pts = np.vstack((xx.ravel(order="F"), yy.ravel(order="F")))
        return self.flagnear_rectangle(pts, opts)

    def nearest(
        self,
        ref: ArrayLike,
        ich: ArrayLike | None = None,
        opts: dict[str, Any] | None = None,
        u: ArrayLike | None = None,
        *,
        max_iterations: int | None = None,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find the nearest point on this chunker to one or more points.

        Chunk indices in ``ich`` and in the returned ``ichn`` are zero-based,
        following the rest of the Python port.
        """

        opts = _legacy_options(opts, "nearest opts")
        _set_option(opts, "nitermax", max_iterations)
        _set_option(opts, "thresh", threshold)
        points = np.asarray(ref, dtype=self.rstor.dtype).reshape(self.dim, -1)
        nref = points.shape[1]
        chunks = np.arange(self.nch) if ich is None else np.asarray(ich, dtype=int).reshape(-1)
        if np.any(chunks < 0) or np.any(chunks >= self.nch):
            raise IndexError("chunk index out of range")

        if u is None:
            _, _, u_arr, _ = lege.exps(self.k)
        else:
            u_arr = np.asarray(u)

        best_dist2 = np.full(nref, np.inf)
        rn = np.zeros((self.dim, nref), dtype=self.rstor.dtype)
        dn = np.zeros_like(rn)
        d2n = np.zeros_like(rn)
        tn = np.zeros(nref, dtype=float)
        ichn = np.full(nref, -1, dtype=int)

        for idx in chunks:
            ti, ri, di, d2i, dist2i = _chunk_nearparam(
                self.r[:, :, idx], points, opts, self.tstor, u_arr
            )
            better = dist2i < best_dist2
            if np.any(better):
                best_dist2[better] = dist2i[better]
                rn[:, better] = ri[:, better]
                dn[:, better] = di[:, better]
                d2n[:, better] = d2i[:, better]
                tn[better] = ti[better]
                ichn[better] = int(idx)

        dist = np.sqrt(best_dist2)
        if np.asarray(ref).reshape(self.dim, -1).shape[1] == 1:
            return rn[:, 0], dn[:, 0], d2n[:, 0], dist[0], tn[0], ichn[0]
        return rn, dn, d2n, dist, tn, ichn

    def min(self) -> np.ndarray:
        if self.nch == 0:
            return np.full(self.dim, np.nan)
        return np.min(np.real(self.r.reshape(self.dim, self.npt, order="F")), axis=1)

    def max(self) -> np.ndarray:
        if self.nch == 0:
            return np.full(self.dim, np.nan)
        return np.max(np.real(self.r.reshape(self.dim, self.npt, order="F")), axis=1)

    def recompute_geometry(self) -> "Chunker":
        self.n = self.normals()
        self.wts = self.weights()
        return self

    def upsample(self, kup: int, sigma: ArrayLike | None = None) -> tuple["Chunker", np.ndarray | None]:
        if kup < self.k:
            raise ValueError("upsampling order must be at least the current order")
        _, _, u, _ = lege.exps(self.k)
        tu, wu, _, vu = lege.exps(kup)
        upmat = vu[:, : self.k] @ u
        out = Chunker(
            ChunkerPref(
                nchmax=max(self.nchmax, self.nch),
                k=kup,
                dim=self.dim,
                nchstor=max(self.nch, 1),
                verttol=self.verttol,
            ),
            tu,
            wu,
        ).addchunk(self.nch)
        out.r = np.einsum("ij,djn->din", upmat, self.r)
        out.d = np.einsum("ij,djn->din", upmat, self.d)
        out.d2 = np.einsum("ij,djn->din", upmat, self.d2)
        out.adj = self.adj
        out.vert = [v.copy() for v in self.vert]
        if self.hasdata:
            out.makedatarows(self.datadim)
            out.data = np.einsum("ij,djn->din", upmat, self.data)
        out.recompute_geometry()

        sigmaup = None
        if sigma is not None:
            sigma_arr = np.asarray(sigma)
            dimsig = sigma_arr.size // (self.k * self.nch)
            sigma_arr = sigma_arr.reshape(dimsig, self.k, self.nch)
            sigmaup = np.einsum("ij,djn->din", upmat, sigma_arr)
        return out, sigmaup

    def split(self, ich: int, frac: float = 0.5, stype: str = "a") -> "Chunker":
        if ich < 0 or ich >= self.nch:
            raise IndexError("chunk index out of range")
        if not (0.0 < frac < 1.0):
            raise ValueError("frac must be between 0 and 1")

        x, _, u, _ = lege.exps(self.k)
        r = self.rstor[:, :, ich]
        d = self.dstor[:, :, ich]
        d2 = self.d2stor[:, :, ich]
        t1 = 2.0 * frac - 1.0

        if stype.lower().startswith("a"):
            dsdt = np.sqrt(np.sum(d**2, axis=0))
            cdsdt = u @ dsdt
            total = float(np.dot(dsdt, self.wstor))
            target = total * frac
            t1 = 0.0
            for _ in range(30):
                ts = -1.0 + (t1 + 1.0) * (x + 1.0) / 2.0
                ws = (t1 + 1.0) * self.wstor / 2.0
                vals = lege.exev(ts, cdsdt)
                err = float(np.dot(vals, ws) - target)
                if abs(err) < 1e-12 * max(total, 1.0):
                    break
                speed = float(np.asarray(lege.exev(np.array([t1]), cdsdt)).reshape(-1)[0])
                t1 -= err / speed
                t1 = min(max(t1, -0.999999999999), 0.999999999999)

        ts1 = -1.0 + (t1 + 1.0) * (x + 1.0) / 2.0
        ts2 = t1 + (1.0 - t1) * (x + 1.0) / 2.0
        h1 = (t1 + 1.0) / 2.0
        h2 = (1.0 - t1) / 2.0

        cr = u @ r.T
        cd = u @ d.T
        cd2 = u @ d2.T
        r1 = lege.exev(ts1, cr).T
        r2 = lege.exev(ts2, cr).T
        d1 = lege.exev(ts1, cd).T
        dnew = lege.exev(ts2, cd).T
        d21 = lege.exev(ts1, cd2).T
        d22 = lege.exev(ts2, cd2).T

        old_nch = self.nch
        right_label = int(self.adjstor[1, ich])
        self.addchunk()
        new_idx = old_nch
        new_label = new_idx + 1

        self.rstor[:, :, ich] = r1
        self.rstor[:, :, new_idx] = r2
        self.dstor[:, :, ich] = d1 * h1
        self.dstor[:, :, new_idx] = dnew * h2
        self.d2stor[:, :, ich] = d21 * h1 * h1
        self.d2stor[:, :, new_idx] = d22 * h2 * h2

        self.adjstor[1, ich] = new_label
        self.adjstor[0, new_idx] = ich + 1
        self.adjstor[1, new_idx] = right_label
        if right_label > 0:
            self.adjstor[0, right_label - 1] = new_label

        if self.hasdata:
            cdata = u @ self.datastor[:, :, ich].T
            self.datastor[:, :, ich] = lege.exev(ts1, cdata).T
            self.datastor[:, :, new_idx] = lege.exev(ts2, cdata).T

        self.recompute_geometry()
        return self

    def refine(
        self,
        opts: dict[str, Any] | None = None,
        *,
        split_chunks: ArrayLike | None = None,
        max_chunk_length: float | None = None,
        level_restrict: str | None = None,
        level_restrict_factor: float | None = None,
        oversample: int | None = None,
        split_type: str | None = None,
        max_chunks: int | None = None,
    ) -> "Chunker":
        """Return a refined copy after selected splits and length balancing.

        Recognized options include ``splitchunks`` for explicit zero-based chunk
        ids, ``maxchunklen`` for arclength-based splitting, ``lvlr``/``lvlrfac``
        for level restriction, ``nover`` for uniform oversampling, and ``stype``
        for arclength versus parameter-space splitting.
        """

        opts = _legacy_options(opts, "refine opts")
        _set_option(opts, "splitchunks", split_chunks)
        _set_option(opts, "maxchunklen", max_chunk_length)
        _set_option(opts, "lvlr", level_restrict)
        _set_option(opts, "lvlrfac", level_restrict_factor)
        _set_option(opts, "nover", oversample)
        _set_option(opts, "stype", split_type)
        _set_option(opts, "nchmax", max_chunks)
        out = self.copy()
        nchmax = int(opts.get("nchmax", out.nchmax))
        if nchmax < out.nch:
            raise ValueError("nchmax must be at least the current number of chunks")
        out.nchmax = nchmax
        stype = str(opts.get("stype", "a"))
        for idx in sorted(np.asarray(opts.get("splitchunks", []), dtype=int).reshape(-1), reverse=True):
            out.split(int(idx), stype=stype)

        maxchunklen = float(opts.get("maxchunklen", np.inf))
        if np.isfinite(maxchunklen):
            maxiter = int(opts.get("maxiter_maxlen", 1000))
            changed = True
            for _ in range(maxiter):
                changed = False
                for idx, length in enumerate(out.chunklen().copy()):
                    if length > maxchunklen:
                        out.split(idx, stype=stype)
                        changed = True
                        break
                if not changed:
                    break
            if changed:
                raise RuntimeError("maximum chunk length refinement did not converge")

        lvlr = str(opts.get("lvlr", "a")).lower()
        if lvlr == "a":
            lvlrfac = float(opts.get("lvlrfac", self.lvlrfacdefault))
            maxiter = int(opts.get("maxiter_lvlr", 1000))
            changed = True
            for _ in range(maxiter):
                changed = False
                lengths = out.chunklen()
                for idx, length in enumerate(lengths.copy()):
                    left = int(out.adj[0, idx])
                    right = int(out.adj[1, idx])
                    left_len = lengths[left - 1] if left > 0 else length
                    right_len = lengths[right - 1] if right > 0 else length
                    if length > lvlrfac * left_len or length > lvlrfac * right_len:
                        out.split(idx, stype=stype)
                        changed = True
                        break
                if not changed:
                    break
            if changed:
                raise RuntimeError("level-restriction refinement did not converge")
        elif lvlr not in {"n", "none"}:
            raise ValueError("lvlr must be 'a' or 'n'")

        for _ in range(int(opts.get("nover", 0))):
            nchold = out.nch
            for idx in range(nchold):
                out.split(idx, stype=stype)
        return out

    def arcresample(self, opts: dict[str, Any] | None = None) -> tuple["Chunker", float]:
        """Reparameterize panel nodes by arc length on each existing chunk."""

        from ..misc import arcparam

        options = {} if opts is None else dict(opts)
        if bool(options.get("mv_bdries", False)):
            sorted_self, info = self.sort()
            components: list[Chunker] = []
            eps = 0.0
            start = 0
            for nchs, closed in zip(info["nchs"], info["ifclosed"]):
                chunks = np.arange(start, start + int(nchs))
                pdata = arcparam.init(sorted_self, chunks)

                def fcurve(s: np.ndarray, pdata=pdata) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                    return arcparam.eval(s, pdata)

                cparams = {
                    _LEGACY_OPTIONS_MARKER: True,
                    "eps": 10.0 * pdata.eps,
                    "ifclosed": bool(closed),
                    "maxchunklen": float(np.max(pdata.plen)),
                    "tb": float(pdata.pstrt[-1]),
                }
                eps = max(eps, float(cparams["eps"]))
                component, _ = chunkerfunc(fcurve, cparams, {"k": self.k, "nchmax": self.nchmax})
                components.append(component)
                start += int(nchs)
            return merge(components), eps

        pdata = arcparam.init(self)
        xs = self.tstor
        legs = lege.pols(xs, self.k - 1)[0].T
        out = self.copy()
        for ich in range(self.nch):
            h = pdata.plen[ich] / 2.0
            out.rstor[:, :, ich] = (legs @ pdata.cr[:, :, ich]).T
            out.dstor[:, :, ich] = (legs @ pdata.cd[:, :, ich]).T * h
            out.d2stor[:, :, ich] = (legs @ pdata.cd2[:, :, ich]).T * h * h
        out.recompute_geometry()
        return out, pdata.eps

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

    def reverse(self) -> "Chunker":
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
    ) -> "Chunker":
        if self.dim != 2:
            raise ValueError("move is implemented for 2D chunkers")
        center0 = np.zeros(2) if r0 is None else np.asarray(r0, dtype=float).reshape(2)
        center1 = np.zeros(2) if r1 is None else np.asarray(r1, dtype=float).reshape(2)
        rot = np.array(
            [[np.cos(trotat), -np.sin(trotat)], [np.sin(trotat), np.cos(trotat)]]
        )
        out = self.copy()
        out.r = scale * np.einsum("ij,jkl->ikl", rot, out.r - center0[:, None, None]) + center1[:, None, None]
        out.d = scale * np.einsum("ij,jkl->ikl", rot, out.d)
        out.d2 = scale * np.einsum("ij,jkl->ikl", rot, out.d2)
        out.n = np.einsum("ij,jkl->ikl", rot, out.n)
        out.wts = out.weights()
        return out

    def rotate(
        self,
        theta: float = 0.0,
        r0: ArrayLike | None = None,
        r1: ArrayLike | None = None,
    ) -> "Chunker":
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
        out.r = np.einsum("ij,jkl->ikl", rot, out.r - center0[:, None, None]) + center1[:, None, None]
        out.d = np.einsum("ij,jkl->ikl", rot, out.d)
        out.d2 = np.einsum("ij,jkl->ikl", rot, out.d2)
        out.n = np.einsum("ij,jkl->ikl", rot, out.n)
        return out

    def reflect(
        self,
        theta: float = 0.0,
        r0: ArrayLike | None = None,
        r1: ArrayLike | None = None,
    ) -> "Chunker":
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
        out.r = np.einsum("ij,jkl->ikl", refmat, out.r - center0[:, None, None]) + center1[:, None, None]
        out.d = np.einsum("ij,jkl->ikl", refmat, out.d)
        out.d2 = np.einsum("ij,jkl->ikl", refmat, out.d2)
        out.n = np.einsum("ij,jkl->ikl", refmat, out.n)
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


_LEGACY_OPTIONS_MARKER = "_chunkie_normalized_geometry_options"


def _legacy_options(opts: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if opts is None:
        return {_LEGACY_OPTIONS_MARKER: True}
    if bool(opts.get(_LEGACY_OPTIONS_MARKER, False)):
        return dict(opts)
    warnings.warn(
        f"{name} dictionaries are deprecated; use keyword-only arguments instead",
        DeprecationWarning,
        stacklevel=3,
    )
    options = dict(opts)
    options[_LEGACY_OPTIONS_MARKER] = True
    return options


def _set_option(options: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        options[key] = value


def _pref_with_order(pref: ChunkerPref | dict[str, Any] | None, order: int | None) -> ChunkerPref:
    p = ChunkerPref.from_any(pref)
    if order is None:
        return p
    return ChunkerPref(p.nchmax, int(order), p.dim, p.nchstor, p.verttol)


def _curve_outputs(fcurve: Callable[[np.ndarray], Any], t: np.ndarray) -> tuple[np.ndarray, ...]:
    raw = fcurve(t)
    if isinstance(raw, tuple):
        outs = raw
    else:
        outs = (raw,)
    return tuple(np.asarray(out, dtype=float).reshape(np.asarray(out).shape[0], -1) for out in outs)


def chunkerfunc(
    fcurve: Callable[[np.ndarray], Any],
    cparams: dict[str, Any] | None = None,
    pref: ChunkerPref | dict[str, Any] | None = None,
    *,
    order: int | None = None,
    closed: bool | None = None,
    interval: tuple[float, float] | None = None,
    tol: float | None = None,
    min_chunks: int | None = None,
    refine: bool | None = None,
    oversample: int | None = None,
    split_points: ArrayLike | None = None,
    max_chunk_length: float | None = None,
    level_restrict: str | None = None,
    level_restrict_factor: float | None = None,
    split_type: str | None = None,
) -> tuple[Chunker, np.ndarray]:
    """Create a chunker for a parameterized curve.

    The implementation follows MATLAB ``chunkerfunc``: initial parameter
    intervals are adaptively split until the curve and speed are spectrally
    resolved, then optional level restriction and oversampling are applied.

    ``fcurve(t)`` must return at least positions with shape ``(dim, len(t))``;
    first and second derivatives may also be returned. Common ``cparams`` are
    ``ta``/``tb`` for the parameter interval, ``ifclosed`` for topology,
    ``eps`` for resolution, ``nchmin`` for a minimum panel count, ``tsplits``
    for forced breakpoints, and ``maxchunklen`` for arclength control.
    """

    cparams = _legacy_options(cparams, "chunkerfunc cparams")
    if interval is not None:
        cparams["ta"], cparams["tb"] = interval
    _set_option(cparams, "ifclosed", closed)
    _set_option(cparams, "eps", tol)
    _set_option(cparams, "nchmin", min_chunks)
    _set_option(cparams, "ifrefine", refine)
    _set_option(cparams, "nover", oversample)
    _set_option(cparams, "tsplits", split_points)
    _set_option(cparams, "maxchunklen", max_chunk_length)
    _set_option(cparams, "lvlr", level_restrict)
    _set_option(cparams, "lvlrfac", level_restrict_factor)
    _set_option(cparams, "stype", split_type)
    p = _pref_with_order(pref, order)

    ta = float(cparams.get("ta", 0.0))
    tb = float(cparams.get("tb", 2.0 * np.pi))
    ifclosed = bool(cparams.get("ifclosed", True))
    nover = int(cparams.get("nover", 0))
    nchmin = int(cparams.get("nchmin", 0))
    tsplits = np.asarray(cparams.get("tsplits", []), dtype=float).reshape(-1)
    eps = float(cparams.get("eps", 1.0e-6))
    ifrefine = bool(cparams.get("ifrefine", True))
    lvlr = str(cparams.get("lvlr", "a")).lower()
    lvlrfac = float(cparams.get("lvlrfac", Chunker.lvlrfacdefault))
    maxchunklen = float(cparams.get("maxchunklen", np.inf))
    chsmall = np.asarray(cparams.get("chsmall", [np.inf, np.inf]), dtype=float).reshape(-1)
    if chsmall.size == 1:
        chsmall = np.repeat(chsmall, 2)
    if chsmall.size != 2:
        raise ValueError("chsmall must be scalar or length 2")

    if tb <= ta:
        raise ValueError("tb must be greater than ta")
    if np.any(tsplits < ta) or np.any(tsplits > tb):
        raise ValueError("tsplits outside interval of definition")

    first = _curve_outputs(fcurve, np.array([ta]))
    dim = first[0].shape[0]
    nout = min(len(first), 3)

    breaks = np.unique(np.concatenate(([ta], tsplits, [tb])))
    breaks.sort()
    if breaks.size < 2:
        raise ValueError("at least one parameter interval is required")

    if nchmin > 0:
        while breaks.size - 1 < nchmin:
            breaks = np.sort(np.concatenate((breaks, 0.5 * (breaks[:-1] + breaks[1:]))))

    if ifrefine:
        breaks = _adaptive_curve_breaks(
            fcurve, breaks, p.k, p.nchmax, dim, nout, eps, maxchunklen, chsmall, ifclosed
        )
    elif np.isfinite(maxchunklen):
        breaks = _maxlen_curve_breaks(fcurve, breaks, p.k, p.nchmax, dim, nout, maxchunklen)

    if lvlr in {"a", "t"}:
        breaks = _level_restrict_curve_breaks(
            fcurve, breaks, p.k, p.nchmax, dim, nout, ifclosed, lvlrfac, lvlr
        )
    elif lvlr not in {"n", "none"}:
        raise ValueError("lvlr must be 'a', 't', or 'n'")

    stype = str(cparams.get("stype", "a")).lower()
    for _ in range(max(nover, 0)):
        breaks = _oversample_curve_breaks(fcurve, breaks, p.k, p.nchmax, dim, nout, stype)

    ab = np.vstack((breaks[:-1], breaks[1:]))
    nch = ab.shape[1]
    if nch > p.nchmax:
        raise ValueError("CHUNKERFUNC: nchmax exceeded")

    p = ChunkerPref(p.nchmax, p.k, dim, max(p.nchstor, min(nch, p.nchmax)), p.verttol)
    chnkr = Chunker(p).addchunk(nch)
    dmat = lege.dermat(chnkr.k)

    for i in range(nch):
        a, b = ab[:, i]
        r, d, d2 = _curve_interval_outputs(fcurve, a, b, chnkr.tstor, dmat, dim, nout)

        chnkr.rstor[:, :, i] = r
        chnkr.dstor[:, :, i] = d
        chnkr.d2stor[:, :, i] = d2

    adjs = np.zeros((2, nch), dtype=int)
    adjs[0] = np.arange(0, nch)
    adjs[1] = np.arange(2, nch + 2)
    if ifclosed:
        adjs[0, 0] = nch
        adjs[1, -1] = 1
    else:
        adjs[0, 0] = -1
        adjs[1, -1] = -1
    chnkr.adj = adjs
    chnkr.recompute_geometry()
    _warn_if_closed_endpoint_mismatch(chnkr, eps, ifclosed, fcurve, ta, tb)
    return chnkr, ab


def _warn_if_closed_endpoint_mismatch(
    chnkr: Chunker,
    eps: float,
    ifclosed: bool,
    fcurve: Callable[[np.ndarray], Any],
    ta: float,
    tb: float,
) -> None:
    if not ifclosed or chnkr.nch == 0:
        return
    endpoint_outputs = _curve_outputs(fcurve, np.array([ta, tb]))
    left = endpoint_outputs[0][:, 0]
    right = endpoint_outputs[0][:, -1]
    bbox = chnkr.max() - chnkr.min()
    scale = float(max(np.max(np.abs(bbox)), 1.0))
    msgbase = "CHUNKERFUNC: "
    if np.linalg.norm(left - right) / scale > eps:
        warnings.warn(
            msgbase
            + "start and end points of curve parameterization are not the same to target precision "
            + "but ifclosed flag is true. Check curve parameterization or if not a closed curve "
            + "set flag appropriately and consider creating a chunkgraph object",
            UserWarning,
            stacklevel=2,
        )
        return
    if len(endpoint_outputs) >= 2:
        left_tangent = endpoint_outputs[1][:, 0]
        right_tangent = endpoint_outputs[1][:, -1]
        left_tangent = left_tangent / np.linalg.norm(left_tangent)
        right_tangent = right_tangent / np.linalg.norm(right_tangent)
    else:
        _, tend = chnkr.chunkends([0, chnkr.nch - 1])
        left_tangent = tend[:, 0, 0]
        right_tangent = tend[:, 1, -1]
    if np.linalg.norm(left_tangent - right_tangent) > eps * chnkr.k:
        warnings.warn(
            msgbase
            + "unit tangent vectors at start and end points of curve are not the same to target precision "
            + "but ifclosed flag is true. Check curve parameterization or if not a closed curve "
            + "set flag appropriately and consider creating a chunkgraph object",
            UserWarning,
            stacklevel=2,
        )


def _curve_interval_outputs(
    fcurve: Callable[[np.ndarray], Any],
    a: float,
    b: float,
    nodes: np.ndarray,
    dmat: np.ndarray,
    dim: int,
    nout: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = (b - a) / 2.0
    ts = a + h * (nodes + 1.0)
    outs = _curve_outputs(fcurve, ts)
    r = outs[0]
    if r.shape != (dim, nodes.size):
        raise ValueError("curve position output has incompatible shape")
    if nout >= 2 and len(outs) >= 2:
        d = outs[1] * h
    else:
        d = r @ dmat.T
    if nout >= 3 and len(outs) >= 3:
        d2 = outs[2] * h * h
    else:
        d2 = d @ dmat.T
    return r, d, d2


def _adaptive_curve_breaks(
    fcurve: Callable[[np.ndarray], Any],
    breaks: np.ndarray,
    k: int,
    nchmax: int,
    dim: int,
    nout: int,
    eps: float,
    maxchunklen: float,
    chsmall: np.ndarray,
    ifclosed: bool,
) -> np.ndarray:
    nodes, weights, u, _ = lege.exps(2 * k)
    dmat = lege.dermat(2 * k)
    for _ in range(max(nchmax, 1)):
        radius = _curve_radius_on_breaks(fcurve, breaks, nodes, dim)
        new_breaks = [float(breaks[0])]
        changed = False
        ninterval = breaks.size - 1
        for idx, (a, b) in enumerate(zip(breaks[:-1], breaks[1:])):
            r, d, d2 = _curve_interval_outputs(fcurve, float(a), float(b), nodes, dmat, dim, nout)
            length = _local_curve_length(d, weights)
            unresolved = _curve_interval_unresolved(r, d, d2, weights, u, k, eps, radius, nout, b - a)
            if np.isfinite(maxchunklen):
                unresolved = unresolved or length > maxchunklen
            if not ifclosed and idx == 0:
                unresolved = unresolved or length > chsmall[0]
            if not ifclosed and idx == ninterval - 1:
                unresolved = unresolved or length > chsmall[1]
            if unresolved:
                new_breaks.append(float(0.5 * (a + b)))
                changed = True
            new_breaks.append(float(b))
            if len(new_breaks) - 1 > nchmax:
                raise ValueError("CHUNKERFUNC: nchmax exceeded. Unable to resolve curve.")
        breaks = np.asarray(new_breaks, dtype=float)
        if not changed:
            return breaks
    raise RuntimeError("adaptive chunkerfunc refinement did not converge")


def _curve_interval_unresolved(
    r: np.ndarray,
    d: np.ndarray,
    d2: np.ndarray,
    weights: np.ndarray,
    u: np.ndarray,
    k: int,
    eps: float,
    radius: float,
    nout: int,
    width: float,
) -> bool:
    speed = np.sqrt(np.sum(np.abs(d) ** 2, axis=0))
    speed_cfs = u @ speed
    low = float(np.sum(np.abs(speed_cfs[:k]) ** 2))
    high = float(np.sum(np.abs(speed_cfs[k:]) ** 2))
    speed_err = np.sqrt(high / max(low, np.finfo(float).eps) / k)
    speed_bad = speed_err > eps if nout >= 2 else speed_err * width > eps * k

    pos_cfs = u @ r.T
    pos_err = np.sqrt(np.max(np.sum(np.abs(pos_cfs[k:, :]) ** 2, axis=0) / k))
    curve_bad = pos_err / max(radius, np.finfo(float).eps) > eps

    curvature_bad = False
    if r.shape[0] == 2:
        zd = d[0] + 1j * d[1]
        zdd = d2[0] + 1j * d2[1]
        denom = np.maximum(np.abs(zd) ** 2, np.finfo(float).eps)
        dkappa = np.imag(zdd * np.conj(zd)) / denom
        curvature_bad = float(np.dot(np.abs(dkappa), weights)) >= (2.0 * np.pi) / 3.0
    return bool(speed_bad or curve_bad or curvature_bad)


def _curve_radius_on_breaks(
    fcurve: Callable[[np.ndarray], Any], breaks: np.ndarray, nodes: np.ndarray, dim: int
) -> float:
    mins = np.full(dim, np.inf)
    maxs = np.full(dim, -np.inf)
    for a, b in zip(breaks[:-1], breaks[1:]):
        ts = float(a) + (float(b) - float(a)) * (nodes + 1.0) / 2.0
        r = _curve_outputs(fcurve, ts)[0]
        mins = np.minimum(mins, np.min(r, axis=1))
        maxs = np.maximum(maxs, np.max(r, axis=1))
    radius = float(np.max(maxs - mins))
    return radius if radius > 0.0 else 1.0


def _maxlen_curve_breaks(
    fcurve: Callable[[np.ndarray], Any],
    breaks: np.ndarray,
    k: int,
    nchmax: int,
    dim: int,
    nout: int,
    maxchunklen: float,
) -> np.ndarray:
    nodes, weights, _, _ = lege.exps(k)
    dmat = lege.dermat(k)
    for _ in range(max(nchmax, 1)):
        new_breaks = [float(breaks[0])]
        changed = False
        for a, b in zip(breaks[:-1], breaks[1:]):
            _, d, _ = _curve_interval_outputs(fcurve, float(a), float(b), nodes, dmat, dim, nout)
            if _local_curve_length(d, weights) > maxchunklen:
                new_breaks.append(float(0.5 * (a + b)))
                changed = True
            new_breaks.append(float(b))
            if len(new_breaks) - 1 > nchmax:
                raise ValueError("CHUNKERFUNC: nchmax exceeded while enforcing maxchunklen")
        breaks = np.asarray(new_breaks, dtype=float)
        if not changed:
            return breaks
    raise RuntimeError("maxchunklen chunkerfunc refinement did not converge")


def _level_restrict_curve_breaks(
    fcurve: Callable[[np.ndarray], Any],
    breaks: np.ndarray,
    k: int,
    nchmax: int,
    dim: int,
    nout: int,
    ifclosed: bool,
    lvlrfac: float,
    lvlr: str,
) -> np.ndarray:
    nodes, weights, _, _ = lege.exps(k)
    dmat = lege.dermat(k)
    for _ in range(1000):
        lengths = np.diff(breaks) if lvlr == "t" else np.array([
            _curve_interval_length(fcurve, float(a), float(b), nodes, weights, dmat, dim, nout)
            for a, b in zip(breaks[:-1], breaks[1:])
        ])
        flags = np.zeros(lengths.size, dtype=bool)
        for idx, length in enumerate(lengths):
            left = lengths[idx - 1] if idx > 0 else (lengths[-1] if ifclosed else length)
            right = lengths[idx + 1] if idx + 1 < lengths.size else (lengths[0] if ifclosed else length)
            flags[idx] = length > lvlrfac * left or length > lvlrfac * right
        if not np.any(flags):
            return breaks
        new_breaks = [float(breaks[0])]
        for idx, (a, b) in enumerate(zip(breaks[:-1], breaks[1:])):
            if flags[idx]:
                new_breaks.append(float(0.5 * (a + b)))
            new_breaks.append(float(b))
            if len(new_breaks) - 1 > nchmax:
                raise ValueError("CHUNKERFUNC: nchmax exceeded during level restriction")
        breaks = np.asarray(new_breaks, dtype=float)
    raise RuntimeError("level-restriction chunkerfunc refinement did not converge")


def _oversample_curve_breaks(
    fcurve: Callable[[np.ndarray], Any],
    breaks: np.ndarray,
    k: int,
    nchmax: int,
    dim: int,
    nout: int,
    stype: str,
) -> np.ndarray:
    nodes, weights, _, _ = lege.exps(k)
    dmat = lege.dermat(k)
    new_breaks = [float(breaks[0])]
    for a, b in zip(breaks[:-1], breaks[1:]):
        if stype.startswith("a"):
            mid = _curve_arclength_midpoint(fcurve, float(a), float(b), nodes, weights, dmat, dim, nout)
        else:
            mid = float(0.5 * (a + b))
        new_breaks.extend((mid, float(b)))
        if len(new_breaks) - 1 > nchmax:
            raise ValueError("CHUNKERFUNC: nchmax exceeded while oversampling")
    return np.asarray(new_breaks, dtype=float)


def _curve_arclength_midpoint(
    fcurve: Callable[[np.ndarray], Any],
    a: float,
    b: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    dmat: np.ndarray,
    dim: int,
    nout: int,
) -> float:
    total = _curve_interval_length(fcurve, a, b, nodes, weights, dmat, dim, nout)
    target = 0.5 * total
    lo = a
    hi = b
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        left = _curve_interval_length(fcurve, a, mid, nodes, weights, dmat, dim, nout)
        if abs(left - target) <= 1e-13 * max(total, 1.0):
            return mid
        if left < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _curve_interval_length(
    fcurve: Callable[[np.ndarray], Any],
    a: float,
    b: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    dmat: np.ndarray,
    dim: int,
    nout: int,
) -> float:
    _, d, _ = _curve_interval_outputs(fcurve, a, b, nodes, dmat, dim, nout)
    return _local_curve_length(d, weights)


def _local_curve_length(d: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(np.sqrt(np.sum(np.abs(d) ** 2, axis=0)), weights))


def chunkerfuncuni(
    fcurve: Callable[[np.ndarray], Any],
    nch: int = 16,
    cparams: dict[str, Any] | None = None,
    pref: ChunkerPref | dict[str, Any] | None = None,
    *,
    order: int | None = None,
    closed: bool | None = None,
    interval: tuple[float, float] | None = None,
) -> Chunker:
    """Create a uniformly panelized chunker from a parametric curve."""

    def uniform_curve(t: np.ndarray) -> Any:
        raw = fcurve(t)
        if isinstance(raw, tuple) and len(raw) > 2:
            # MATLAB chunkerfuncuni computes local second derivatives spectrally.
            return raw[:2]
        return raw

    params = _legacy_options(cparams, "chunkerfuncuni cparams")
    if interval is not None:
        params["ta"], params["tb"] = interval
    _set_option(params, "ifclosed", closed)
    nch = int(nch)
    ta = float(params.get("ta", 0.0))
    tb = float(params.get("tb", 2.0 * np.pi))
    ifclosed = bool(params.get("ifclosed", True))
    params = {_LEGACY_OPTIONS_MARKER: True, "ta": ta, "tb": tb, "ifclosed": ifclosed}
    params["tsplits"] = np.linspace(ta, tb, nch + 1)[1:-1]
    params["ifrefine"] = False
    params["lvlr"] = "n"
    params["nover"] = 0
    chnkr, _ = chunkerfunc(uniform_curve, params, pref, order=order)
    return chnkr


def chunkerfit(
    xy: ArrayLike,
    opts: dict[str, Any] | None = None,
    *,
    closed: bool | None = None,
    split_at_points: bool | None = None,
    tol: float | None = None,
    order: int | None = None,
    pref: ChunkerPref | dict[str, Any] | None = None,
) -> Chunker:
    """Create a chunker by fitting a cubic spline through 2D points."""

    from scipy.interpolate import CubicSpline

    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[0] != 2:
        raise ValueError("Points must be specified as a 2xN matrix")
    options = _legacy_options(opts, "chunkerfit opts")
    _set_option(options, "ifclosed", closed)
    _set_option(options, "splitatpoints", split_at_points)
    if tol is not None:
        options.setdefault("cparams", {})["eps"] = tol
    if pref is not None:
        options["pref"] = pref
    if order is not None:
        p0 = ChunkerPref.from_any(options.get("pref", None))
        options["pref"] = ChunkerPref(p0.nchmax, int(order), p0.dim, p0.nchstor, p0.verttol)
    method = str(options.get("method", "spline")).lower()
    if method != "spline":
        raise ValueError(f"Unsupported method {method!r}")

    ifclosed = bool(options.get("ifclosed", True))
    pts = points
    if ifclosed and np.linalg.norm(points[:, 0] - points[:, -1]) > 1e-14:
        pts = np.column_stack((points, points[:, 0]))
    if pts.shape[1] < 3:
        raise ValueError("chunkerfit requires at least three points")

    seglen = np.sqrt(np.sum(np.diff(pts, axis=1) ** 2, axis=0))
    if np.any(seglen <= 0.0):
        raise ValueError("consecutive fit points must be distinct")
    t = np.concatenate(([0.0], np.cumsum(seglen)))

    bc_type = "periodic" if ifclosed else "not-a-knot"
    splx = CubicSpline(t, pts[0], bc_type=bc_type)
    sply = CubicSpline(t, pts[1], bc_type=bc_type)

    def splinefunc(tt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tt_arr = np.asarray(tt)
        return (
            np.vstack((splx(tt_arr), sply(tt_arr))),
            np.vstack((splx(tt_arr, 1), sply(tt_arr, 1))),
            np.vstack((splx(tt_arr, 2), sply(tt_arr, 2))),
        )

    cparams = dict(options.get("cparams", {}))
    cparams[_LEGACY_OPTIONS_MARKER] = True
    cparams["ifclosed"] = ifclosed
    cparams["ta"] = float(t[0])
    cparams["tb"] = float(t[-1])
    if bool(options.get("splitatpoints", False)):
        cparams["tsplits"] = t[1:-1]
    chnkr, _ = chunkerfunc(splinefunc, cparams, options.get("pref", None))
    # MATLAB chunkerfit's local ppdiff helper leaves fitted second derivatives
    # zero in the returned chunker; keep that observable behavior for parity.
    chnkr.d2 = np.zeros_like(chnkr.d2)
    return chnkr


def chunkerpoly(
    verts: ArrayLike,
    cparams: dict[str, Any] | None = None,
    pref: ChunkerPref | dict[str, Any] | None = None,
    edgevals: ArrayLike | None = None,
    *,
    order: int | None = None,
    closed: bool | None = None,
    dyadic: bool | None = None,
    depth: int | None = None,
    rounded: bool | None = None,
    widths: ArrayLike | None = None,
) -> Chunker:
    """Create a chunker for a true polygon or open polyline.

    By default each polygon edge is one straight chunk. With
    ``cparams={"dyadic": True, "depth": ...}``, edges are refined
    geometrically near corners for non-smooth boundary-integral workflows. With
    ``cparams={"rounded": True}``, corners are trimmed and replaced by
    lightweight quadratic panels. The full MATLAB Gaussian smoother is richer,
    but the Python paths preserve the same high-level workflow and edge-data
    propagation.
    """

    cparams = _legacy_options(cparams, "chunkerpoly cparams")
    _set_option(cparams, "ifclosed", closed)
    _set_option(cparams, "dyadic", dyadic)
    _set_option(cparams, "depth", depth)
    _set_option(cparams, "rounded", rounded)
    _set_option(cparams, "widths", widths)
    rounded = bool(cparams.get("rounded", False))

    vertices = np.asarray(verts, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 2 or vertices.shape[1] < 2:
        raise ValueError("verts must have shape (dim, nverts) with dim > 1")
    dim, nv = vertices.shape
    ifclosed = bool(cparams.get("ifclosed", True))
    p = _pref_with_order(pref, order)
    p = ChunkerPref(p.nchmax, p.k, dim, max(p.nchstor, nv), p.verttol)

    if rounded:
        return _rounded_chunkerpoly(vertices, cparams, p, edgevals)
    if bool(cparams.get("dyadic", "depth" in cparams)):
        return _dyadic_chunkerpoly(vertices, cparams, p, edgevals)

    if ifclosed:
        starts = vertices
        ends = np.column_stack((vertices[:, 1:], vertices[:, 0]))
    else:
        starts = vertices[:, :-1]
        ends = vertices[:, 1:]
    nedge = starts.shape[1]

    if nedge > p.nchmax:
        raise ValueError("too many polygon edges for nchmax")
    chnkr = Chunker(p).addchunk(nedge)

    edge_data = None
    if edgevals is not None:
        edge_data = np.asarray(edgevals, dtype=float)
        if edge_data.size % nedge != 0:
            raise ValueError("number of edge values should be multiple of number of edges")
        edge_data = edge_data.reshape(edge_data.size // nedge, nedge)
        chnkr.makedatarows(edge_data.shape[0])

    t01 = (chnkr.tstor + 1.0) / 2.0
    for idx in range(nedge):
        start = starts[:, idx]
        end = ends[:, idx]
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            raise ValueError("polygon edges must have positive length")
        tangent = delta / length
        h = length / 2.0
        chnkr.rstor[:, :, idx] = start[:, None] + delta[:, None] * t01[None, :]
        chnkr.dstor[:, :, idx] = tangent[:, None] * h
        chnkr.d2stor[:, :, idx] = 0.0
        if edge_data is not None:
            chnkr.datastor[:, :, idx] = edge_data[:, idx][:, None]

    adjs = np.zeros((2, nedge), dtype=int)
    adjs[0] = np.arange(0, nedge)
    adjs[1] = np.arange(2, nedge + 2)
    if ifclosed:
        adjs[0, 0] = nedge
        adjs[1, -1] = 1
    else:
        adjs[0, 0] = -1
        adjs[1, -1] = -1
    chnkr.adj = adjs
    chnkr.recompute_geometry()
    return chnkr


def chunkerpoints(
    src: ArrayLike | dict[str, ArrayLike],
    opts: dict[str, Any] | None = None,
    *,
    closed: bool | None = None,
) -> Chunker:
    """Create a chunker from panel node values.

    ``src`` may be either a ``(dim, k, nch)`` position array or a mapping
    with ``r`` and optional matching ``d``/``d2`` arrays, mirroring MATLAB
    ``chunkerpoints``.
    """

    opts = _legacy_options(opts, "chunkerpoints opts")
    _set_option(opts, "ifclosed", closed)
    d_arr = None
    d2_arr = None
    if isinstance(src, dict):
        if "r" not in src:
            raise ValueError("missing field r in chunkerpoints")
        r_arr = np.asarray(src["r"])
        if "d" in src and np.asarray(src["d"]).shape == r_arr.shape:
            d_arr = np.asarray(src["d"], dtype=r_arr.dtype)
        if "d2" in src and np.asarray(src["d2"]).shape == r_arr.shape:
            d2_arr = np.asarray(src["d2"], dtype=r_arr.dtype)
    else:
        r_arr = np.asarray(src)

    if r_arr.ndim != 3:
        raise ValueError("chunkerpoints expects r with shape (dim, k, nch)")
    dim, k, nch = r_arr.shape
    if nch <= 0:
        raise ValueError("chunkerpoints requires at least one chunk")

    pref = ChunkerPref(dim=dim, k=k, nchstor=nch, nchmax=max(nch, 1))
    chnkr = Chunker(pref).addchunk(nch)
    dmat = lege.dermat(k)
    chnkr.r = r_arr
    if d_arr is None:
        chnkr.d = np.einsum("dkn,jk->djn", r_arr, dmat)
    else:
        chnkr.d = d_arr
    if d2_arr is None:
        chnkr.d2 = np.einsum("dkn,jk->djn", chnkr.d, dmat)
    else:
        chnkr.d2 = d2_arr

    adjs = np.zeros((2, nch), dtype=int)
    adjs[0] = np.arange(0, nch)
    adjs[1] = np.arange(2, nch + 2)
    if bool(opts.get("ifclosed", True)):
        adjs[0, 0] = nch
        adjs[1, -1] = 1
    else:
        adjs[0, 0] = -1
        adjs[1, -1] = -1
    chnkr.adj = adjs
    chnkr.recompute_geometry()
    return chnkr


def merge(
    chnkrs: ArrayLike | list[Chunker] | tuple[Chunker, ...],
    pref: ChunkerPref | dict[str, Any] | None = None,
) -> Chunker:
    """Combine chunkers of the same dimension and order.

    The merged chunker keeps each input component's local adjacency, shifting
    positive neighbor labels by the accumulated chunk offset. This is the
    geometry path used when scalar kernels are applied to a ``ChunkGraph`` or
    to an explicit list of chunkers.
    """

    if isinstance(chnkrs, Chunker):
        items = [chnkrs]
    elif isinstance(chnkrs, (list, tuple)):
        items = list(chnkrs)
    else:
        items = list(np.ravel(chnkrs))
    if not items:
        return Chunker(pref)
    if not all(isinstance(item, Chunker) for item in items):
        raise TypeError("input must contain chunker objects")

    first = items[0]
    total_nch = sum(item.nch for item in items)
    p = ChunkerPref.from_any(pref)
    p = ChunkerPref(
        nchmax=max(p.nchmax, total_nch),
        k=first.k,
        dim=first.dim,
        nchstor=max(p.nchstor, total_nch),
        verttol=p.verttol,
    )
    out = Chunker(p, first.tstor, first.wstor).addchunk(total_nch)

    offset = 0
    for item in items:
        if item.dim != first.dim or item.k != first.k:
            raise ValueError("chunkers to merge must have the same dimension and order")
        sl = slice(offset, offset + item.nch)
        out.rstor[:, :, sl] = item.r
        out.dstor[:, :, sl] = item.d
        out.d2stor[:, :, sl] = item.d2
        out.nstor[:, :, sl] = item.n
        out.wtsstor[:, sl] = item.wts
        adj = item.adj.copy()
        adj[adj > 0] += offset
        out.adjstor[:, sl] = adj
        offset += item.nch

    max_data = max((item.datadim for item in items), default=0)
    if max_data > 0:
        out.makedatarows(max_data)
        offset = 0
        for item in items:
            if item.hasdata and item.datadim > 0:
                out.datastor[: item.datadim, :, offset : offset + item.nch] = item.data
            offset += item.nch
    return out


def _bernstein_rectangle_info(chnkr: Chunker, rho: float) -> np.ndarray:
    """Return MATLAB-style rectangle tests for Bernstein ellipse images."""

    ells = _bernstein_ellipse_images(chnkr, rho)
    _, dc, _ = chnkr.exps()
    p0 = _legendre_values(np.array([0.0]), chnkr.k - 1).reshape(chnkr.k)
    d0 = np.einsum("k,dkn->dn", p0, dc)
    d0_norm = np.sqrt(np.sum(d0**2, axis=0))
    d1s = d0 / d0_norm[None, :]
    d2s = np.vstack((d1s[1], -d1s[0]))

    d1c = np.einsum("dmn,dn->mn", ells, d1s)
    d2c = np.einsum("dmn,dn->mn", ells, d2s)

    rectinfo = np.zeros((2, 4, chnkr.nch))
    rectinfo[:, 0, :] = d1s
    rectinfo[:, 1, :] = d2s
    rectinfo[0, 2, :] = np.min(d1c, axis=0)
    rectinfo[1, 2, :] = np.max(d1c, axis=0)
    rectinfo[0, 3, :] = np.min(d2c, axis=0)
    rectinfo[1, 3, :] = np.max(d2c, axis=0)
    return rectinfo


def _bernstein_ellipse_images(chnkr: Chunker, rho: float) -> np.ndarray:
    nth = max(2 * chnkr.nch, 20)
    theta = np.linspace(0.0, 2.0 * np.pi, nth + 1)[:-1]
    zrho = rho * np.exp(1j * theta)
    zell = (zrho + 1.0 / zrho) / 2.0
    zpols = _legendre_values(zell, chnkr.k - 1).T
    rc, _, _ = chnkr.exps()
    zcoef = rc[0] + 1j * rc[1]
    ell = zpols @ zcoef
    return np.stack((ell.real, ell.imag), axis=0)


def _legendre_values(xs: ArrayLike, degree: int) -> np.ndarray:
    xs_arr = np.asarray(xs)
    flat = xs_arr.reshape(-1)
    vals = np.zeros((degree + 1, flat.size), dtype=np.result_type(xs_arr, float))
    vals[0] = 1.0
    if degree >= 1:
        vals[1] = flat
    for k in range(1, degree):
        vals[k + 1] = ((2 * k + 1) * flat * vals[k] - k * vals[k - 1]) / (k + 1)
    return vals.reshape((degree + 1,) + xs_arr.shape)


def _remap_adjacency(adjs: np.ndarray, inds: np.ndarray) -> np.ndarray:
    inverse = {old + 1: new + 1 for new, old in enumerate(inds)}
    out = adjs.copy()
    for old_label, new_label in inverse.items():
        out[adjs == old_label] = new_label
    return out
