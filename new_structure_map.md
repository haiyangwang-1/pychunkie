# New Structure Map

This document maps the old MATLAB-shaped Python source layout to the refactored
layout. Old paths refer to the Python tree as it existed immediately before the
`chnk` namespace was removed. MATLAB reference paths such as `+chnk/+quadggq`
remain useful for parity notes, but they are no longer mirrored as Python
package names.

## Summary

The old `src/chunkie/chnk/` package mixed geometry, kernels, quadrature,
numerics, and acceleration helpers because it closely followed MATLAB
`+chnk`. The refactor groups Python code by responsibility:

- `chunkie.geometry`: curve constructors and geometric predicates.
- `chunkie.numerics`: arclength parametrization, smoother helpers, and special
  scalar functions.
- `chunkie.quadrature`: native dense assembly, GGQ, adaptive correction, panel
  product quadrature, and RCIP helpers.
- `chunkie.kernels`: concrete physics kernel families used by
  `chunkie.kernel`.
- `chunkie.acceleration`: FLAM callback and proxy helpers.

The `chunkie.chnk` package has no replacement compatibility wrapper. Import the
new responsibility-based package directly.

## Source Path Map

| Old path | New path | Notes |
| --- | --- | --- |
| `src/chunkie/chnk/__init__.py` | removed | The `chunkie.chnk` namespace was deleted. |
| `src/chunkie/chnk/arcparam.py` | `src/chunkie/numerics/arcparam.py` | Arclength parametrization helpers. |
| `src/chunkie/chnk/smoother.py` | `src/chunkie/numerics/smoother.py` | Lightweight polygon smoothing helpers. |
| `src/chunkie/chnk/spcl.py` | `src/chunkie/numerics/special.py` | Special scalar helper functions. |
| `src/chunkie/chnk/curves.py` | `src/chunkie/geometry/curves.py` | Curve constructors such as `linefunc`, `fsine`, and `fpara`. |
| `src/chunkie/chnk/geometry.py` | `src/chunkie/geometry/predicates.py` | Point, normal, curvature, near, and self-interaction predicates. |
| `src/chunkie/chnk/quadnative.py` | `src/chunkie/quadrature/native.py` | Native smooth dense assembly. |
| `src/chunkie/chnk/quadggq.py` | `src/chunkie/quadrature/ggq.py` | Generalized Gaussian quadrature setup and assembly. |
| `src/chunkie/chnk/quadadap.py` | `src/chunkie/quadrature/adaptive.py` | Adaptive close-evaluation corrections. |
| `src/chunkie/chnk/pquad.py` | `src/chunkie/quadrature/panel.py` | Panel product quadrature helpers. |
| `src/chunkie/chnk/rcip.py` | `src/chunkie/quadrature/rcip.py` | RCIP and corner-compression helpers. |
| `src/chunkie/chnk/lap2d.py` | `src/chunkie/kernels/laplace.py` | Laplace kernel formulas and FMM wiring. |
| `src/chunkie/chnk/helm2d.py` | `src/chunkie/kernels/helmholtz.py` | 2D Helmholtz kernel formulas and FMM wiring. |
| `src/chunkie/chnk/helm1d.py` | `src/chunkie/kernels/helmholtz_1d.py` | 1D Helmholtz helpers. |
| `src/chunkie/chnk/biharm2d.py` | `src/chunkie/kernels/biharmonic.py` | Biharmonic kernel formulas. |
| `src/chunkie/chnk/stok2d.py` | `src/chunkie/kernels/stokes.py` | Stokes kernel formulas and FMM wiring. |
| `src/chunkie/chnk/elast2d.py` | `src/chunkie/kernels/elasticity.py` | Elasticity kernel formulas. |
| `src/chunkie/chnk/flam.py` | `src/chunkie/acceleration/flam.py` | FLAM index callbacks and proxy helpers. |

## Retained Top-Level Modules

| Path | Status after refactor | Role |
| --- | --- | --- |
| `src/chunkie/__init__.py` | retained | Small public package facade for chunkers, domains, kernels, and operators. |
| `src/chunkie/chunker.py` | retained | Core `Chunker` data structure and chunker constructors. |
| `src/chunkie/chunkgraph.py` | retained | Multi-edge and region-aware chunker graph helpers. |
| `src/chunkie/domain.py` | retained | Domain and region utilities. |
| `src/chunkie/kernel.py` | retained | Public `Kernel` object and `kernel(...)` factory facade; concrete implementations now come from `chunkie.kernels`. |
| `src/chunkie/operators.py` | retained | Dense, FMM, FLAM, and special-quadrature operator orchestration. |
| `src/chunkie/lege/` | retained | Legendre utilities. |
| `src/chunkie/data/quadggq/` | retained | Packaged GGQ table data used by `chunkie.quadrature.ggq`. |

## Import Migration Map

| Old import | New import |
| --- | --- |
| `from chunkie.chnk import arcparam` | `from chunkie.numerics import arcparam` |
| `from chunkie.chnk import smoother` | `from chunkie.numerics import smoother` |
| `from chunkie.chnk import spcl` | `from chunkie.numerics import special` |
| `from chunkie.chnk import curves` | `from chunkie.geometry import curves` |
| `from chunkie.chnk import geometry` | `from chunkie.geometry import predicates` |
| `from chunkie.chnk import quadnative` | `from chunkie.quadrature import native as quadnative` |
| `from chunkie.chnk import quadggq` | `from chunkie.quadrature import ggq as quadggq` |
| `from chunkie.chnk import quadadap` | `from chunkie.quadrature import adaptive as quadadap` |
| `from chunkie.chnk import pquad` | `from chunkie.quadrature import panel as pquad` |
| `from chunkie.chnk import rcip` | `from chunkie.quadrature import rcip` |
| `from chunkie.chnk import lap2d` | `from chunkie.kernels import laplace as lap2d` |
| `from chunkie.chnk import helm2d` | `from chunkie.kernels import helmholtz as helm2d` |
| `from chunkie.chnk import helm1d` | `from chunkie.kernels import helmholtz_1d as helm1d` |
| `from chunkie.chnk import biharm2d` | `from chunkie.kernels import biharmonic as biharm2d` |
| `from chunkie.chnk import stok2d` | `from chunkie.kernels import stokes as stok2d` |
| `from chunkie.chnk import elast2d` | `from chunkie.kernels import elasticity as elast2d` |
| `from chunkie.chnk import flam` | `from chunkie.acceleration import flam` |

## Internal Dependency Map

| Refactored package | Main users |
| --- | --- |
| `chunkie.geometry` | `chunker.py`, `chunkgraph.py`, domain tests, geometry parity tests. |
| `chunkie.numerics` | `chunker.py`, smoother tests, arclength parity tests, special-function tests. |
| `chunkie.quadrature` | `operators.py`, quadrature tests, RCIP examples and parity tests. |
| `chunkie.kernels` | `kernel.py`, kernel tests, operator tests, MATLAB parity tests. |
| `chunkie.acceleration` | `operators.py`, FLAM tests, devtools parity tests. |

## Notes For Future Refactors

- Keep `kernel.py` as the public kernel factory layer. Move only concrete
  physics formulas into `chunkie.kernels`.
- Keep `operators.py` as the assembly/application coordinator. Splitting it
  further should be driven by a real ownership boundary, not by file size alone.
- Keep MATLAB reference names in parity docs and fixture descriptions where they
  identify upstream behavior, even though Python imports now use
  responsibility-based package names.
