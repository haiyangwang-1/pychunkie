"""Two-dimensional Stokes kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from chunkie.operators import PointInfo, pointinfo


def kern(
    mu: float,
    srcinfo: PointInfo | dict | ArrayLike,
    targinfo: PointInfo | dict | ArrayLike,
    kind: str,
    coefs: ArrayLike | None = None,
) -> np.ndarray:
    src = pointinfo(srcinfo)
    targ = pointinfo(targinfo)
    typ = kind.lower()
    rx = targ.r[0, :, None] - src.r[0, None, :]
    ry = targ.r[1, :, None] - src.r[1, None, :]
    r2 = rx**2 + ry**2
    nt, ns = rx.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        if typ in {"svel", "s", "single"}:
            r = np.sqrt(r2)
            log1r = np.log(1.0 / r)
            kxx = (rx**2 / r2 + log1r) / (4.0 * np.pi * mu)
            kyy = (ry**2 / r2 + log1r) / (4.0 * np.pi * mu)
            kxy = (rx * ry / r2) / (4.0 * np.pi * mu)
            return _interleave_2x2(kxx, kxy, kxy, kyy, nt, ns)
        if typ == "spres":
            kx = rx / (2.0 * np.pi * r2)
            ky = ry / (2.0 * np.pi * r2)
            return _interleave_1x2(kx, ky, nt, ns)
        if typ == "strac":
            _require(targ.n, "target normals")
            r4 = r2**2
            rn = rx * targ.n[0, :, None] + ry * targ.n[1, :, None]
            kxx = -rx**2 * rn / (np.pi * r4)
            kyy = -ry**2 * rn / (np.pi * r4)
            kxy = -rx * ry * rn / (np.pi * r4)
            return _interleave_2x2(kxx, kxy, kxy, kyy, nt, ns)
        if typ in {"dvel", "d", "double"}:
            _require(src.n, "source normals")
            r4 = r2**2
            rn = rx * src.n[0, None, :] + ry * src.n[1, None, :]
            kxx = rx**2 * rn / (np.pi * r4)
            kyy = ry**2 * rn / (np.pi * r4)
            kxy = rx * ry * rn / (np.pi * r4)
            return _interleave_2x2(kxx, kxy, kxy, kyy, nt, ns)
        if typ == "dpres":
            _require(src.n, "source normals")
            r4 = r2**2
            rn = rx * src.n[0, None, :] + ry * src.n[1, None, :]
            kx = mu * (-src.n[0, None, :] / r2 + 2.0 * rx * rn / r4) / np.pi
            ky = mu * (-src.n[1, None, :] / r2 + 2.0 * ry * rn / r4) / np.pi
            return _interleave_1x2(kx, ky, nt, ns)
        if typ == "dtrac":
            _require(src.n, "source normals")
            _require(targ.n, "target normals")
            r4 = r2**2
            r6 = r2**3
            nsx = src.n[0, None, :]
            nsy = src.n[1, None, :]
            ntx = targ.n[0, :, None]
            nty = targ.n[1, :, None]
            rns = rx * nsx + ry * nsy
            rnt = rx * ntx + ry * nty
            nn = ntx * nsx + nty * nsy
            kxx = mu / np.pi * (-8 * rx**2 * rnt * rns / r6 + (rx * ntx * rns + rx**2 * nn + rnt * rns + nsx * rx * rnt) / r4 + ntx * nsx / r2)
            kyy = mu / np.pi * (-8 * ry**2 * rnt * rns / r6 + (ry * nty * rns + ry**2 * nn + rnt * rns + nsy * ry * rnt) / r4 + nty * nsy / r2)
            kxy = mu / np.pi * (-8 * rx * ry * rnt * rns / r6 + (rx * nty * rns + rx * ry * nn + nsx * ry * rnt) / r4 + ntx * nsy / r2)
            kyx = mu / np.pi * (-8 * ry * rx * rnt * rns / r6 + (ry * ntx * rns + ry * rx * nn + nsy * rx * rnt) / r4 + nty * nsx / r2)
            return _interleave_2x2(kxx, kxy, kyx, kyy, nt, ns)
        if typ in {"sgrad", "sg"}:
            r4inv = 1.0 / r2**2 / (4.0 * np.pi * mu)
            return _interleave_grad(
                rx * (ry**2 - rx**2) * r4inv,
                ry * (ry**2 - rx**2) * r4inv,
                (-3 * rx**2 * ry - ry**3) * r4inv,
                rx * (rx**2 - ry**2) * r4inv,
                ry * (ry**2 - rx**2) * r4inv,
                (-3 * rx * ry**2 - rx**3) * r4inv,
                rx * (rx**2 - ry**2) * r4inv,
                ry * (rx**2 - ry**2) * r4inv,
                nt,
                ns,
            )
        if typ in {"dgrad", "dg"}:
            _require(src.n, "source normals")
            r4 = r2**2
            r6 = r2**3
            nsx = src.n[0, None, :]
            nsy = src.n[1, None, :]
            rn = rx * nsx + ry * nsy
            op = 1.0 / np.pi
            return _interleave_grad(
                (rn * rx / r4 + rx * rn / r4 + rx**2 * nsx / r4 - 4 * rx**3 * rn / r6) * op,
                (rn * ry / r4 + rx * ry * nsx / r4 - 4 * rx**2 * ry * rn / r6) * op,
                (rx**2 * nsy / r4 - 4 * rx**2 * ry * rn / r6) * op,
                (rx * rn / r4 + rx * ry * nsy / r4 - 4 * rx * ry**2 * rn / r6) * op,
                (ry * rn / r4 + rx * ry * nsx / r4 - 4 * rx**2 * ry * rn / r6) * op,
                (ry**2 * nsx / r4 - 4 * rx * ry**2 * rn / r6) * op,
                (rn * rx / r4 + rx * ry * nsy / r4 - 4 * rx * ry**2 * rn / r6) * op,
                (rn * ry / r4 + ry * rn / r4 + ry**2 * nsy / r4 - 4 * ry**3 * rn / r6) * op,
                nt,
                ns,
            )
        if typ in {"c", "combined"}:
            c = np.ones(2) if coefs is None else np.asarray(coefs)
            return c[0] * kern(mu, src, targ, "d") + c[1] * kern(mu, src, targ, "s")
    raise ValueError(f"Unknown Stokes kernel type {kind!r}.")


def _interleave_2x2(kxx, kxy, kyx, kyy, nt, ns):
    out = np.zeros((2 * nt, 2 * ns), dtype=np.result_type(kxx, kxy, kyx, kyy))
    out[0::2, 0::2] = kxx
    out[0::2, 1::2] = kxy
    out[1::2, 0::2] = kyx
    out[1::2, 1::2] = kyy
    return out


def _interleave_1x2(kx, ky, nt, ns):
    out = np.zeros((nt, 2 * ns), dtype=np.result_type(kx, ky))
    out[:, 0::2] = kx
    out[:, 1::2] = ky
    return out


def _interleave_grad(a, b, c, d, e, f, g, h, nt, ns):
    out = np.zeros((4 * nt, 2 * ns), dtype=np.result_type(a, b, c, d, e, f, g, h))
    out[0::4, 0::2] = a
    out[0::4, 1::2] = b
    out[1::4, 0::2] = c
    out[1::4, 1::2] = d
    out[2::4, 0::2] = e
    out[2::4, 1::2] = f
    out[3::4, 0::2] = g
    out[3::4, 1::2] = h
    return out


def _require(value: object, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} are required")
