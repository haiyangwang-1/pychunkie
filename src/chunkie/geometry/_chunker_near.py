"""Private chunkernearmixin methods for :class:`chunkie.geometry.Chunker`."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie._layout import as_boundary_point_matrix

from .. import lege
from ._chunker_options import _legacy_options, _set_option
from ._nearest import _bernstein_rectangle_info
from ._nearest import chunk_nearparam as _chunk_nearparam


class ChunkerNearMixin:
    def flagnear(
        self,
        points: ArrayLike,
        options: dict[str, Any] | None = None,
        *,
        fac: float | None = None,
    ) -> np.ndarray:
        options = _legacy_options(options, "flagnear options")
        _set_option(options, "fac", fac)
        fac = float(options.get("fac", 1.0))
        points_arr = np.asarray(points, dtype=float).reshape(self.dim, -1)
        flags = np.zeros((points_arr.shape[1], self.nch), dtype=bool)
        lens = self.chunklen() * fac
        for ich in range(self.nch):
            diff = points_arr[:, :, None] - self.r[:, :, ich][:, None, :]
            dists = np.sqrt(np.sum(diff**2, axis=0))
            flags[:, ich] = np.any(dists < lens[ich], axis=1)
        return flags

    def flagnear_rectangle(
        self,
        points: ArrayLike,
        options: dict[str, Any] | None = None,
        *,
        rho: float | None = None,
    ) -> np.ndarray:
        options = _legacy_options(options, "flagnear_rectangle options")
        _set_option(options, "rho", rho)
        if self.dim != 2:
            raise ValueError("flagnear_rectangle is implemented for 2D chunkers")
        rho = float(options.get("rho", 1.8))
        rectinfo = _bernstein_rectangle_info(self, rho)
        points_arr = np.asarray(points, dtype=float).reshape(2, -1)
        flags = np.zeros((points_arr.shape[1], self.nch), dtype=bool)
        for ich in range(self.nch):
            d1 = points_arr.T @ rectinfo[:, 0, ich]
            d2 = points_arr.T @ rectinfo[:, 1, ich]
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
        options: dict[str, Any] | None = None,
        *,
        rho: float | None = None,
    ) -> np.ndarray:
        """Flag a Cartesian target grid, returning ``(len(y), len(x), nch)``."""

        options = _legacy_options(options, "flagnear_rectangle_grid options")
        _set_option(options, "rho", rho)
        xx, yy = np.meshgrid(
            np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)
        )
        pts = np.vstack((xx.ravel(), yy.ravel()))
        flags = self.flagnear_rectangle(pts, options)
        return flags.reshape(xx.shape + (flags.shape[1],))

    def nearest(
        self,
        points: ArrayLike,
        chunks: ArrayLike | None = None,
        options: dict[str, Any] | None = None,
        node_parameters: ArrayLike | None = None,
        *,
        max_iterations: int | None = None,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find the nearest point on this chunker to one or more points.

        Chunk indices in ``chunks`` and in the returned ``ichn`` are zero-based,
        following the rest of the Python port.
        """

        options = _legacy_options(options, "nearest options")
        _set_option(options, "nitermax", max_iterations)
        _set_option(options, "thresh", threshold)
        points_arr = np.asarray(points, dtype=self.rstor.dtype).reshape(self.dim, -1)
        nref = points_arr.shape[1]
        chunk_ids = (
            np.arange(self.nch) if chunks is None else np.asarray(chunks, dtype=int).reshape(-1)
        )
        if np.any(chunk_ids < 0) or np.any(chunk_ids >= self.nch):
            raise IndexError("chunk index out of range")

        if node_parameters is None:
            _, _, u_arr, _ = lege.exps(self.k)
        else:
            u_arr = np.asarray(node_parameters)

        best_dist2 = np.full(nref, np.inf)
        rn = np.zeros((self.dim, nref), dtype=self.rstor.dtype)
        dn = np.zeros_like(rn)
        d2n = np.zeros_like(rn)
        tn = np.zeros(nref, dtype=float)
        ichn = np.full(nref, -1, dtype=int)

        for idx in chunk_ids:
            ti, ri, di, d2i, dist2i = _chunk_nearparam(
                self.r[:, :, idx], points_arr, options, self.tstor, u_arr
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
        if np.asarray(points).reshape(self.dim, -1).shape[1] == 1:
            return rn[:, 0], dn[:, 0], d2n[:, 0], dist[0], tn[0], ichn[0]
        return rn, dn, d2n, dist, tn, ichn

    def min(self) -> np.ndarray:
        if self.nch == 0:
            return np.full(self.dim, np.nan)
        return np.min(
            np.real(as_boundary_point_matrix(self.r, self.dim, self.npt, name="positions")), axis=1
        )

    def max(self) -> np.ndarray:
        if self.nch == 0:
            return np.full(self.dim, np.nan)
        return np.max(
            np.real(as_boundary_point_matrix(self.r, self.dim, self.npt, name="positions")), axis=1
        )
