# New Structure Map

This document maps the MATLAB-shaped Python layout to the current refactored
layout. MATLAB reference paths such as `+chnk/+quadggq` remain useful for parity
notes, but Python code no longer mirrors those package names.

## Summary

- `chunkie.geometry`: `Chunker`, `ChunkerPref`, `ChunkGraph`, chunker
  constructors, polygon helpers, `PointInfo`, curve helpers, and domain
  helpers.
- `chunkie.misc`: arclength parametrization, smoother helpers, and
  `absconvgauss`.
- `chunkie.quadrature`: native dense assembly, GGQ, adaptive correction, and
  panel product quadrature.
- `chunkie.rcip`: corner-compression and recursive local-refinement helpers.
- `chunkie.kernels`: concrete kernel families plus the `Kernel` wrapper and
  `kernel(...)` factory.
- `chunkie.acceleration`: FLAM callback and proxy helpers.
- `chunkie.operators`: dense/FMM/FLAM operator orchestration.

The deleted `chunkie.chnk` package and the old low-level module paths do not
have compatibility wrappers.

## Source Path Map

| Old path | Current path | Notes |
| --- | --- | --- |
| `src/chunkie/chnk/__init__.py` | removed | The `chunkie.chnk` namespace was deleted. |
| `src/chunkie/chunker.py` | `src/chunkie/geometry/chunker.py` | `Chunker`, `ChunkerPref`, chunker constructors, merge helpers, and keyword-only option APIs. |
| `src/chunkie/chunkgraph.py` | `src/chunkie/geometry/chunkgraph.py` | `ChunkGraph`, graph constructors, region helpers, graph refinement, and graph near-flag methods. |
| `src/chunkie/domain.py` | `src/chunkie/geometry/domain.py` | Top-level curve, region, and hyperoctree domain helpers. |
| `src/chunkie/_chunker_polygon.py` | `src/chunkie/geometry/_chunker_polygon.py` | Private polygon construction helpers used by `chunkerpoly`. |
| `src/chunkie/geometry/predicates.py` | removed | Dead public helpers were deleted. The nearest-panel routine is now private in `geometry/_nearest.py`. |
| `src/chunkie/operators.py::PointInfo` | `src/chunkie/geometry/pointinfo.py` | Public construction is through `PointInfo.from_chunker`, `from_points`, and `from_mapping`. |
| `src/chunkie/numerics/arcparam.py` | `src/chunkie/misc/arcparam.py` | Arclength parametrization helpers. |
| `src/chunkie/numerics/smoother.py` | `src/chunkie/misc/smoother.py` | Lightweight polygon smoothing helpers. |
| `src/chunkie/numerics/special.py` | `src/chunkie/misc/absconvgauss.py` | Special scalar helper module. |
| `src/chunkie/kernel.py` | `src/chunkie/kernels/factory.py` | `Kernel`, `kernel(...)`, algebra helpers, and FMM dispatch wiring. |
| `src/chunkie/chnk/curves.py` | `src/chunkie/geometry/curves.py` | Curve constructors such as `linefunc`, `fsine`, and `fpara`. |
| `src/chunkie/chnk/quadnative.py` | `src/chunkie/quadrature/native.py` | Native smooth dense assembly. |
| `src/chunkie/chnk/quadggq.py` | `src/chunkie/quadrature/ggq.py` | Generalized Gaussian quadrature setup and assembly. |
| `src/chunkie/chnk/quadadap.py` | `src/chunkie/quadrature/adaptive.py` | Adaptive close-evaluation corrections. |
| `src/chunkie/chnk/pquad.py` | `src/chunkie/quadrature/panel.py` | Panel product quadrature helpers. |
| `src/chunkie/chnk/rcip.py` | `src/chunkie/rcip/core.py`, `src/chunkie/rcip/algebra.py`, `src/chunkie/rcip/types.py` | RCIP and corner-compression helpers. |
| `src/chunkie/chnk/lap2d.py` | `src/chunkie/kernels/laplace.py` | Laplace formulas and FMM wiring. |
| `src/chunkie/chnk/helm2d.py` | `src/chunkie/kernels/helmholtz.py` | 2D Helmholtz formulas and FMM wiring. |
| `src/chunkie/chnk/helm1d.py` | `src/chunkie/kernels/helmholtz_1d.py` | 1D Helmholtz helpers. |
| `src/chunkie/chnk/biharm2d.py` | `src/chunkie/kernels/biharmonic.py` | Biharmonic formulas. |
| `src/chunkie/chnk/stok2d.py` | `src/chunkie/kernels/stokes.py` | Stokes formulas and FMM wiring. |
| `src/chunkie/chnk/elast2d.py` | `src/chunkie/kernels/elasticity.py` | Elasticity formulas. |
| `src/chunkie/chnk/flam.py` | `src/chunkie/acceleration/flam.py` | FLAM index callbacks and proxy helpers. |

## Import Migration Map

| Old import | Current import |
| --- | --- |
| `from chunkie import pointinfo` | `from chunkie import PointInfo`; use `PointInfo.from_chunker(...)`, `from_points(...)`, or `from_mapping(...)` |
| `from chunkie import Chunker, ChunkGraph, chunkerfunc` | unchanged top-level facade, backed by `chunkie.geometry` |
| `from chunkie.chunker import Chunker` | `from chunkie.geometry import Chunker` |
| `from chunkie.chunkgraph import ChunkGraph` | `from chunkie.geometry import ChunkGraph` |
| `from chunkie import ellipse, pointinregion` | unchanged top-level facade, backed by `chunkie.geometry.domain` |
| `from chunkie.geometry import flagself` | removed |
| `from chunkie.geometry import perp, normal2d, curvature2d` | removed; use direct NumPy expressions or `Chunker.normals()` / `Chunker.signed_curvature()` where applicable |
| `from chunkie.geometry import chunk_nearparam` | removed public API; use `Chunker.nearest(...)` |
| `from chunkie.numerics import arcparam` | `from chunkie.misc import arcparam` |
| `from chunkie.numerics import smoother` | `from chunkie.misc import smoother` |
| `from chunkie.numerics import special` | `from chunkie.misc import absconvgauss` |
| `from chunkie.kernel import Kernel, kernel` | `from chunkie.kernels import Kernel, kernel` |
| `from chunkie.chnk import curves` | `from chunkie.geometry import curves` |
| `from chunkie.chnk import quadnative` | `from chunkie.quadrature import native as quadnative` |
| `from chunkie.chnk import quadggq` | `from chunkie.quadrature import ggq as quadggq` |
| `from chunkie.chnk import quadadap` | `from chunkie.quadrature import adaptive as quadadap` |
| `from chunkie.chnk import pquad` | `from chunkie.quadrature import panel as pquad` |
| `from chunkie.chnk import rcip` | `from chunkie import rcip` |
| `from chunkie.chnk import lap2d` | `from chunkie.kernels import laplace as lap2d` |
| `from chunkie.chnk import helm2d` | `from chunkie.kernels import helmholtz as helm2d` |
| `from chunkie.chnk import helm1d` | `from chunkie.kernels import helmholtz_1d as helm1d` |
| `from chunkie.chnk import biharm2d` | `from chunkie.kernels import biharmonic as biharm2d` |
| `from chunkie.chnk import stok2d` | `from chunkie.kernels import stokes as stok2d` |
| `from chunkie.chnk import elast2d` | `from chunkie.kernels import elasticity as elast2d` |
| `from chunkie.chnk import flam` | `from chunkie.acceleration import flam` |

## Options API

Public chunker and operator entry points now accept keyword-only options with
defaults. Old option dictionaries are still accepted temporarily and emit
`DeprecationWarning`, then normalize through the same internal option path.

Examples:

- `chunkerfunc(curve, order=16, closed=True, tol=1e-6, min_chunks=8)`
- `chunkerpoly(vertices, order=16, closed=True, dyadic=True, depth=3)`
- `boundary.refine(oversample=1, level_restrict="a")`
- `chunkermat(boundary, kernel_obj, acceleration="fmm", tol=1e-12)`
- `chunkerkerneval(boundary, kernel_obj, density, targets, force_adaptive=True, use_panel_quadrature=False)`

## Internal Dependency Map

| Refactored package | Main users |
| --- | --- |
| `chunkie.geometry` | operators, domain helpers, quadrature, geometry tests, parity tests. |
| `chunkie.misc` | chunker arclength resampling, smoother tests, arclength parity tests, special-function tests. |
| `chunkie.quadrature` | operators, quadrature tests, and special-quadrature parity tests. |
| `chunkie.rcip` | operators, RCIP tests, RCIP parity tests, and nonsmooth chunkgraph workflows. |
| `chunkie.kernels` | operators, kernel tests, operator tests, MATLAB parity tests. |
| `chunkie.acceleration` | operators, FLAM tests, devtools parity tests. |

## Notes For Future Refactors

- Keep `operators.py` as the assembly/application coordinator unless a real
  ownership boundary appears.
- Keep concrete kernel math in `chunkie.kernels`; no JIT work was introduced in
  this structure pass.
- Keep MATLAB reference names in parity docs and fixture descriptions where they
  identify upstream behavior, even though Python imports now use
  responsibility-based package names.
