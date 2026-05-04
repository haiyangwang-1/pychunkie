"""Chunked curve data structure mirroring MATLAB ``@chunker``."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
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

    def arclengthdens(self) -> np.ndarray:
        return np.sqrt(np.sum(self.d**2, axis=0))

    def arclengthder(self, u: ArrayLike) -> np.ndarray:
        dmat = lege.dermat(self.k)
        vals = np.asarray(u).reshape(self.k, self.nch)
        return (dmat @ vals) / self.arclengthdens()

    def arclengthfun(self) -> np.ndarray:
        aint = lege.intmat(self.k)[0]
        s = aint @ self.arclengthdens()
        starts = np.concatenate(([0.0], np.cumsum(self.chunklen()[:-1])))
        return s + starts[None, :]

    def chunklen(self, ich: ArrayLike | None = None) -> np.ndarray:
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
        wts = self.wts.reshape(-1)
        return np.ones((self.npt, 1)) @ wts[None, :]

    def normonesmat(self) -> np.ndarray:
        normals = self.n.reshape(-1)
        wts2 = (np.repeat(self.wts.reshape(-1), self.dim) * normals)
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

    def flagnear(self, pts: ArrayLike, opts: dict[str, Any] | None = None) -> np.ndarray:
        opts = {} if opts is None else dict(opts)
        fac = float(opts.get("fac", 1.0))
        points = np.asarray(pts, dtype=float).reshape(self.dim, -1)
        flags = np.zeros((points.shape[1], self.nch), dtype=bool)
        lens = self.chunklen() * fac
        for ich in range(self.nch):
            diff = points[:, :, None] - self.r[:, :, ich][:, None, :]
            dists = np.sqrt(np.sum(diff**2, axis=0))
            flags[:, ich] = np.any(dists < lens[ich], axis=1)
        return flags

    def nearest(
        self,
        ref: ArrayLike,
        ich: ArrayLike | None = None,
        opts: dict[str, Any] | None = None,
        u: ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find the nearest point on this chunker to one or more points.

        Chunk indices in ``ich`` and in the returned ``ichn`` are zero-based,
        following the rest of the Python port.
        """

        from .chnk.geometry import chunk_nearparam

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
            ti, ri, di, d2i, dist2i = chunk_nearparam(
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
        return np.min(np.real(self.r.reshape(self.dim, self.npt)), axis=1)

    def max(self) -> np.ndarray:
        if self.nch == 0:
            return np.full(self.dim, np.nan)
        return np.max(np.real(self.r.reshape(self.dim, self.npt)), axis=1)

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
                speed = float(lege.exev(np.array([t1]), cdsdt))
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

    def refine(self, opts: dict[str, Any] | None = None) -> "Chunker":
        opts = {} if opts is None else dict(opts)
        out = self.copy()
        stype = str(opts.get("stype", "a"))
        for idx in sorted(np.asarray(opts.get("splitchunks", []), dtype=int).reshape(-1), reverse=True):
            out.split(int(idx), stype=stype)

        maxchunklen = float(opts.get("maxchunklen", np.inf))
        if np.isfinite(maxchunklen):
            changed = True
            while changed:
                changed = False
                for idx, length in enumerate(out.chunklen().copy()):
                    if length > maxchunklen:
                        out.split(idx, stype=stype)
                        changed = True
                        break

        for _ in range(int(opts.get("nover", 0))):
            nchold = out.nch
            for idx in range(nchold):
                out.split(idx, stype=stype)
        return out

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
) -> tuple[Chunker, np.ndarray]:
    """Create a chunker for a parameterized curve.

    This is the fixed-layout first port of MATLAB ``chunkerfunc``. It honors
    ``ta``, ``tb``, ``ifclosed``, ``tsplits``, ``nover``, and ``nchmin``.
    Adaptive refinement options are intentionally deferred.
    """

    cparams = {} if cparams is None else dict(cparams)
    p = ChunkerPref.from_any(pref)

    ta = float(cparams.get("ta", 0.0))
    tb = float(cparams.get("tb", 2.0 * np.pi))
    ifclosed = bool(cparams.get("ifclosed", True))
    nover = int(cparams.get("nover", 0))
    nchmin = int(cparams.get("nchmin", 0))
    tsplits = np.asarray(cparams.get("tsplits", []), dtype=float).reshape(-1)

    if tb <= ta:
        raise ValueError("tb must be greater than ta")
    if np.any(tsplits < ta) or np.any(tsplits > tb):
        raise ValueError("tsplits outside interval of definition")

    breaks = np.unique(np.concatenate(([ta], tsplits, [tb])))
    breaks.sort()
    if breaks.size < 2:
        raise ValueError("at least one parameter interval is required")

    if nchmin > 0:
        while breaks.size - 1 < nchmin:
            breaks = np.sort(np.concatenate((breaks, 0.5 * (breaks[:-1] + breaks[1:]))))

    for _ in range(max(nover, 0)):
        breaks = np.sort(np.concatenate((breaks, 0.5 * (breaks[:-1] + breaks[1:]))))

    ab = np.vstack((breaks[:-1], breaks[1:]))
    nch = ab.shape[1]
    if nch > p.nchmax:
        raise ValueError("CHUNKERFUNC: nchmax exceeded")

    first = _curve_outputs(fcurve, np.array([ta]))
    dim = first[0].shape[0]
    p = ChunkerPref(p.nchmax, p.k, dim, max(p.nchstor, min(nch, p.nchmax)), p.verttol)
    chnkr = Chunker(p).addchunk(nch)
    dmat = lege.dermat(chnkr.k)

    for i in range(nch):
        a, b = ab[:, i]
        h = (b - a) / 2.0
        ts = a + h * (chnkr.tstor + 1.0)
        outs = _curve_outputs(fcurve, ts)
        r = outs[0]
        if r.shape != (dim, chnkr.k):
            raise ValueError("curve position output has incompatible shape")

        if len(outs) >= 2:
            d = outs[1] * h
        else:
            d = r @ dmat.T
        if len(outs) >= 3:
            d2 = outs[2] * h * h
        else:
            d2 = d @ dmat.T

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
    return chnkr, ab


def chunkerpoly(
    verts: ArrayLike,
    cparams: dict[str, Any] | None = None,
    pref: ChunkerPref | dict[str, Any] | None = None,
    edgevals: ArrayLike | None = None,
) -> Chunker:
    """Create a chunker for a true polygon or open polyline.

    Rounded corners and dyadic corner refinement from MATLAB ``chunkerpoly``
    are deferred; this baseline builds one panel per edge.
    """

    cparams = {} if cparams is None else dict(cparams)
    rounded = bool(cparams.get("rounded", False))
    if rounded:
        raise NotImplementedError("rounded chunkerpoly corners are not implemented yet")

    vertices = np.asarray(verts, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 2 or vertices.shape[1] < 2:
        raise ValueError("verts must have shape (dim, nverts) with dim > 1")
    dim, nv = vertices.shape
    ifclosed = bool(cparams.get("ifclosed", True))
    p = ChunkerPref.from_any(pref)
    p = ChunkerPref(p.nchmax, p.k, dim, max(p.nchstor, nv), p.verttol)

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


def chunkerpoints(src: ArrayLike | dict[str, ArrayLike], opts: dict[str, Any] | None = None) -> Chunker:
    """Create a chunker from panel node values.

    ``src`` may be either a ``(dim, k, nch)`` position array or a mapping
    with ``r`` and optional matching ``d``/``d2`` arrays, mirroring MATLAB
    ``chunkerpoints``.
    """

    opts = {} if opts is None else dict(opts)
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


def _remap_adjacency(adjs: np.ndarray, inds: np.ndarray) -> np.ndarray:
    inverse = {old + 1: new + 1 for new, old in enumerate(inds)}
    out = adjs.copy()
    for old_label, new_label in inverse.items():
        out[adjs == old_label] = new_label
    return out
