"""Private refinement/resampling methods for :class:`chunkie.geometry.Chunker`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .. import lege
from ._chunker_options import _LEGACY_OPTIONS_MARKER, _legacy_options, _set_option
from ._chunker_pref import ChunkerPref

if TYPE_CHECKING:
    from ._chunker_class import Chunker


class ChunkerRefineMixin:
    def recompute_geometry(self) -> Chunker:
        self.n = self.normals()
        self.wts = self.weights()
        return self

    def upsample(
        self, kup: int, sigma: ArrayLike | None = None
    ) -> tuple[Chunker, np.ndarray | None]:
        if kup < self.k:
            raise ValueError("upsampling order must be at least the current order")
        _, _, u, _ = lege.exps(self.k)
        tu, wu, _, vu = lege.exps(kup)
        upmat = vu[:, : self.k] @ u
        out = type(self)(
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

    def split(self, ich: int, frac: float = 0.5, stype: str = "a") -> Chunker:
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
        options: dict[str, Any] | None = None,
        *,
        split_chunks: ArrayLike | None = None,
        max_chunk_length: float | None = None,
        level_restrict: str | None = None,
        level_restrict_factor: float | None = None,
        oversample: int | None = None,
        split_type: str | None = None,
        max_chunks: int | None = None,
    ) -> Chunker:
        """Return a refined copy after selected splits and length balancing.

        Recognized options include ``splitchunks`` for explicit zero-based chunk
        ids, ``maxchunklen`` for arclength-based splitting, ``lvlr``/``lvlrfac``
        for level restriction, ``nover`` for uniform oversampling, and ``stype``
        for arclength versus parameter-space splitting.
        """

        options = _legacy_options(options, "refine options")
        _set_option(options, "splitchunks", split_chunks)
        _set_option(options, "maxchunklen", max_chunk_length)
        _set_option(options, "lvlr", level_restrict)
        _set_option(options, "lvlrfac", level_restrict_factor)
        _set_option(options, "nover", oversample)
        _set_option(options, "stype", split_type)
        _set_option(options, "nchmax", max_chunks)
        out = self.copy()
        nchmax = int(options.get("nchmax", out.nchmax))
        if nchmax < out.nch:
            raise ValueError("nchmax must be at least the current number of chunks")
        out.nchmax = nchmax
        stype = str(options.get("stype", "a"))
        for idx in sorted(
            np.asarray(options.get("splitchunks", []), dtype=int).reshape(-1), reverse=True
        ):
            out.split(int(idx), stype=stype)

        maxchunklen = float(options.get("maxchunklen", np.inf))
        if np.isfinite(maxchunklen):
            maxiter = int(options.get("maxiter_maxlen", 1000))
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

        lvlr = str(options.get("lvlr", "a")).lower()
        if lvlr == "a":
            lvlrfac = float(options.get("lvlrfac", self.lvlrfacdefault))
            maxiter = int(options.get("maxiter_lvlr", 1000))
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

        for _ in range(int(options.get("nover", 0))):
            nchold = out.nch
            for idx in range(nchold):
                out.split(idx, stype=stype)
        return out

    def arcresample(self, options: dict[str, Any] | None = None) -> tuple[Chunker, float]:
        """Reparameterize panel nodes by arc length on each existing chunk."""

        from ..misc import arcparam

        options = {} if options is None else dict(options)
        if bool(options.get("mv_bdries", False)):
            sorted_self, info = self.sort()
            components: list[Chunker] = []
            eps = 0.0
            start = 0
            for nchs, closed in zip(info["nchs"], info["ifclosed"], strict=False):
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
                from ._chunker_func import chunkerfunc
                from ._chunker_points import merge

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
