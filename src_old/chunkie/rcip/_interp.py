"""RCIP saved-density interpolation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie._layout import as_boundary_field_matrix, as_boundary_point_matrix, as_boundary_vector

from ..geometry.chunker import Chunker
from .types import RCIPSaved


def rhohatInterp(rhohat: ArrayLike, rcipsav: RCIPSaved | dict[str, Any], ndepth: int | None = None):
    """Interpolate a compressed RCIP density back through saved levels."""

    rho = as_boundary_vector(rhohat, name="rhohat")
    if isinstance(rcipsav, dict):
        nsub = int(rcipsav.get("nsub", 0))
        savedepth = int(rcipsav.get("savedepth", nsub))
        locals_ = rcipsav.get("chnkrlocals", [])
        rlist = rcipsav.get("R", [])
        matlist = rcipsav.get("MAT", [])
        pbc = np.asarray(rcipsav["Pbc"])
        star_s = np.sort(np.asarray(rcipsav["starS"], dtype=int).reshape(-1))
        circ_s = np.sort(np.asarray(rcipsav["circS"], dtype=int).reshape(-1))
        star_l1 = np.sort(np.asarray(rcipsav["starL1"], dtype=int).reshape(-1))
        circ_l1 = np.sort(np.asarray(rcipsav["circL1"], dtype=int).reshape(-1))
        nedge = int(rcipsav["nedge"])
        ileftright = np.asarray(rcipsav.get("ileftright", np.ones(nedge)), dtype=int).reshape(-1)
    else:
        nsub = rcipsav.nsub
        savedepth = int(rcipsav.savedepth)
        locals_ = [] if rcipsav.chnkrlocals is None else rcipsav.chnkrlocals
        rlist = [] if rcipsav.R is None else rcipsav.R
        matlist = [] if rcipsav.MAT is None else rcipsav.MAT
        pbc = rcipsav.Pbc
        star_s = np.sort(rcipsav.starS)
        circ_s = np.sort(rcipsav.circS)
        star_l1 = np.sort(rcipsav.starL1)
        circ_l1 = np.sort(rcipsav.circL1)
        nedge = rcipsav.nedge
        ileftright = np.ones(nedge, dtype=int) if rcipsav.ileftright is None else rcipsav.ileftright

    if nsub <= 0:
        return [rho.copy()], [None], [None]

    depth = nsub if ndepth is None else min(int(ndepth), nsub)
    if depth > savedepth:
        raise ValueError("requested interpolation depth exceeds saved RCIP depth")
    if len(rlist) < depth + 1 or len(matlist) < depth or len(locals_) < depth:
        raise ValueError("rcipsav does not contain enough saved recursion data")

    nrho = circ_s.size + star_s.size
    if rho.size % nrho != 0:
        raise ValueError("rhohat has incompatible size for RCIP saved data")
    ndens = rho.size // nrho
    rhohat0 = as_boundary_field_matrix(rho, nrho, ndens, name="rhohat")

    circ_s_edges = _split_edge_indices(circ_s, nedge)
    star_s_edges = _split_edge_indices(star_s, nedge)
    circ_l1_edges = _split_edge_indices(circ_l1, nedge)
    star_l1_edges = _split_edge_indices(star_l1, nedge)

    cl = locals_[-1]
    wt = as_boundary_vector(cl.wts, name="weights")
    rhohat_interpolation = [rhohat0[idx, :].copy() for idx in circ_s_edges]
    srcinfo = [_pointinfo_subset(cl, idx) for idx in circ_l1_edges]
    wts = [wt[idx].copy() for idx in circ_l1_edges]

    r0 = rlist[-1]
    for idepth in range(1, depth + 1):
        r1 = rlist[-idepth - 1]
        mat = matlist[-idepth]
        rhotemp = np.linalg.solve(r0, rhohat0)
        rhohat0 = r1 @ (pbc @ rhotemp[star_s, :] - mat @ rhohat0[circ_s, :])
        if idepth == depth:
            for iedge in range(nedge):
                order = (
                    np.concatenate((circ_s_edges[iedge], star_s_edges[iedge]))
                    if int(ileftright[iedge]) == 1
                    else np.concatenate((star_s_edges[iedge], circ_s_edges[iedge]))
                )
                rhohat_interpolation[iedge] = np.vstack(
                    (rhohat_interpolation[iedge], rhohat0[order, :])
                )
                srcinfo[iedge] = _pointinfo_append(
                    srcinfo[iedge], _pointinfo_subset(cl, star_l1_edges[iedge])
                )
                wts[iedge] = np.concatenate((wts[iedge], wt[star_l1_edges[iedge]]))
        else:
            cl = locals_[-idepth - 1]
            wt = as_boundary_vector(cl.wts, name="weights")
            for iedge in range(nedge):
                rhohat_interpolation[iedge] = np.vstack(
                    (rhohat_interpolation[iedge], rhohat0[circ_s_edges[iedge], :])
                )
                srcinfo[iedge] = _pointinfo_append(
                    srcinfo[iedge], _pointinfo_subset(cl, circ_l1_edges[iedge])
                )
                wts[iedge] = np.concatenate((wts[iedge], wt[circ_l1_edges[iedge]]))
        r0 = r1

    if ndens == 1:
        rhohat_interpolation = [vals[:, 0] for vals in rhohat_interpolation]
    return rhohat_interpolation, srcinfo, wts


def _split_edge_indices(indices: np.ndarray, nedge: int) -> list[np.ndarray]:
    if indices.size % int(nedge) != 0:
        raise ValueError("RCIP saved indices are not evenly split by edge")
    per_edge = indices.size // int(nedge)
    return [indices[i * per_edge : (i + 1) * per_edge] for i in range(int(nedge))]


def _pointinfo_subset(chunker: Chunker, indices: np.ndarray) -> Any:
    from ..operators import PointInfo

    return PointInfo(
        r=as_boundary_point_matrix(chunker.r, chunker.dim, chunker.npt, name="positions")[
            :, indices
        ],
        d=as_boundary_point_matrix(chunker.d, chunker.dim, chunker.npt, name="derivatives")[
            :, indices
        ],
        d2=as_boundary_point_matrix(
            chunker.d2, chunker.dim, chunker.npt, name="second derivatives"
        )[:, indices],
        n=as_boundary_point_matrix(chunker.n, chunker.dim, chunker.npt, name="normals")[:, indices],
    )


def _pointinfo_append(left: Any, right: Any) -> Any:
    from ..operators import PointInfo

    return PointInfo(
        r=np.column_stack((left.r, right.r)),
        d=np.column_stack((left.d, right.d)),
        d2=np.column_stack((left.d2, right.d2)),
        n=np.column_stack((left.n, right.n)),
    )
