"""Private chunkergeometrymixin methods for :class:`chunkie.geometry.Chunker`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie._layout import as_boundary_vector, boundary_component_weights

from .. import lege
from ._chunker_adjacency import _remap_adjacency

if TYPE_CHECKING:
    from ._chunker_class import Chunker


class ChunkerGeometryMixin:
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
        wts = as_boundary_vector(self.wts, name="weights")
        return np.ones((self.npt, 1)) @ wts[None, :]

    def normonesmat(self) -> np.ndarray:
        normals = as_boundary_vector(self.n, name="normals")
        wts2 = boundary_component_weights(self.wts, self.dim) * normals
        return normals[:, None] @ wts2[None, :]

    def centroids(self) -> np.ndarray:
        return np.sum(self.r * self.wstor[None, :, None], axis=1) / 2.0

    def datares(self, options: dict[str, Any] | None = None) -> np.ndarray:
        """Check whether selected data rows are Legendre-resolved per chunk."""

        options = {} if options is None else dict(options)
        if not self.hasdata or self.datadim == 0:
            return np.zeros((0, self.nch), dtype=bool)

        idata = np.asarray(options.get("idata", np.arange(self.datadim)), dtype=int).reshape(-1)
        if np.any(idata < 0) or np.any(idata >= self.datadim):
            raise IndexError("data row index out of range")

        ncoeff = int(options.get("ncoeff", np.floor((self.k + 0.1) / 2.0)))
        ncoeff = min(max(ncoeff, 1), self.k)
        pleg = options.get("pleg", 1)
        tol = float(options.get("tol", 1.0e-6))
        pscale = float(options.get("pscale", 0.0))
        rel = bool(options.get("rel", False))

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

    def sort(self) -> tuple[Chunker, dict[str, Any]]:
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
