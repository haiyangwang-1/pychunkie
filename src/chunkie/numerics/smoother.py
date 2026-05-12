"""Lightweight polygon smoother helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .. import lege
from ..chunker import chunkerpoly


@dataclass
class UniformMesh:
    verts: np.ndarray
    centroids: np.ndarray
    lengths: np.ndarray
    face_normals: np.ndarray
    pseudo_normals: np.ndarray


@dataclass
class SmoothMesh:
    r: np.ndarray
    n: np.ndarray
    pseudo_normals: np.ndarray
    wts: np.ndarray


def get_umesh(verts: ArrayLike) -> UniformMesh:
    """Return the edge mesh used by MATLAB ``chnk.smoother`` helpers."""

    vertices = np.asarray(verts, dtype=float)
    if vertices.shape[0] != 2 or vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError("verts must have shape (2, nverts) with nverts >= 3")
    verts_ext = np.column_stack((vertices, vertices[:, 0]))
    d = verts_ext[:, 1:] - verts_ext[:, :-1]
    lengths = np.sqrt(np.sum(d**2, axis=0))
    if np.any(lengths <= 0.0):
        raise ValueError("polygon edges must have positive length")
    centroids = 0.5 * (verts_ext[:, 1:] + verts_ext[:, :-1])
    face_normals = np.vstack((d[1] / lengths, -d[0] / lengths))
    face_ext = np.column_stack((face_normals[:, -1], face_normals))
    pseudo_normals = 0.5 * (face_ext[:, 1:] + face_ext[:, :-1])
    pseudo_normals /= np.sqrt(np.sum(pseudo_normals**2, axis=0))[None, :]
    return UniformMesh(vertices, centroids, lengths, face_normals, pseudo_normals)


def get_mesh(umesh: UniformMesh, nchs: ArrayLike, k: int) -> SmoothMesh:
    """Sample a uniform smoother mesh with Legendre panels on each edge."""

    nchs_arr = np.asarray(nchs, dtype=int).reshape(-1)
    if nchs_arr.size == 1:
        nchs_arr = np.full(umesh.verts.shape[1], int(nchs_arr[0]))
    if nchs_arr.size != umesh.verts.shape[1] or np.any(nchs_arr <= 0):
        raise ValueError("nchs must be a positive scalar or one value per edge")
    x, w, _, _ = lege.exps(int(k))
    verts_ext = np.column_stack((umesh.verts, umesh.verts[:, 0]))
    pseudo_ext = np.column_stack((umesh.pseudo_normals, umesh.pseudo_normals[:, 0]))

    rs: list[np.ndarray] = []
    ns: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    for iedge, nch in enumerate(nchs_arr):
        xext, wext = _panel_nodes(x, w, int(nch))
        r = verts_ext[:, iedge, None] + (verts_ext[:, iedge + 1] - verts_ext[:, iedge])[:, None] * xext[None, :]
        pnorm = pseudo_ext[:, iedge, None] + (pseudo_ext[:, iedge + 1] - pseudo_ext[:, iedge])[:, None] * xext[None, :]
        rs.append(r)
        ns.append(np.repeat(umesh.face_normals[:, iedge, None], xext.size, axis=1))
        ps.append(pnorm)
        ws.append(wext * umesh.lengths[iedge])
    return SmoothMesh(np.hstack(rs), np.hstack(ns), np.hstack(ps), np.concatenate(ws))


def smooth(verts: ArrayLike, opts: dict[str, Any] | None = None):
    """Build a rounded chunker from polygon vertices.

    This is a dependency-light smoother workflow compatible with the
    rounded ``chunkerpoly`` path. It returns ``(chnkr, err, err_by_pt)``
    when ``return_error`` is true in ``opts``; otherwise it returns only
    the chunker, like MATLAB's first output.
    """

    options = {} if opts is None else dict(opts)
    k = int(options.get("k", 16))
    cparams = {
        "rounded": True,
        "ifclosed": options.get("ifclosed", True),
        "autowidthsfac": options.get("autowidthsfac", 0.1),
    }
    if "widths" in options:
        cparams["widths"] = options["widths"]
    chnkr = chunkerpoly(verts, cparams, {"k": k})
    err_by_pt = np.zeros(chnkr.npt)
    err = 0.0
    if bool(options.get("return_error", False)):
        return chnkr, err, err_by_pt
    return chnkr


smooth_curve = smooth
smooth_curve2 = smooth
smooth_curve3 = smooth


def _panel_nodes(x: np.ndarray, w: np.ndarray, nch: int) -> tuple[np.ndarray, np.ndarray]:
    tabs = np.linspace(0.0, 1.0, nch + 1)
    xs: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    for a, b in zip(tabs[:-1], tabs[1:]):
        xs.append((x + 1.0) * (b - a) / 2.0 + a)
        ws.append(w * (b - a) / 2.0)
    return np.concatenate(xs), np.concatenate(ws)
