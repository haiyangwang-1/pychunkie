# Python Test Suite Summary

Status: reset for the clean rewrite.

Update trigger: update this file when tests are added, removed, renamed,
parametrized, or when the behavior covered by existing tests changes
meaningfully.

## Current Snapshot

- Previous test-suite summary is archived in
  `docs_old/docs/python-test-suite-summary.md`.
- Active tests currently cover the rewrite bootstrap and first low-level APIs.
- Verification snapshot: `uv run pytest -q` on 2026-05-17,
  `66 passed` in 1.91 seconds.

## Test Organization Target

- `tests/geometry/`: `Chunker`, point views, `ChunkGraph`, near geometry, and regions.
- `tests/kernels/`: formulas, selector metadata, singularity metadata, and algebra.
- `tests/quadrature/`: dense panel rules, Helsing-Ojala, GGQ, and adaptive fallback.
- `tests/rcip/`: local compression and interpolation.
- `tests/system/`: density layout, assembly, traces, solves, and evaluation.
- `tests/backends/`: FMM and FLAM comparisons against dense references.

## Active Tests

- `tests/test_api_smoke.py`: top-level rewrite API and removal of old operator
  names from the active facade.
- `tests/geometry/test_chunker.py`: panel-major `Chunker` storage, normals,
  point maps, polygon construction, near-panel flags, `ChunkGraph` merged
  point views, boundary parts, and single-boundary region classification.
- `tests/geometry/test_chunker_metrics.py`: migrated geometry metric and
  transform behavior for panel lengths, total length, signed area, tangents,
  curvature, translation, scaling, affine maps, rotation, and reflection.
- `tests/geometry/test_curve_constructors.py`: migrated curve-constructor
  behavior for closed analytic curves, open curves with free ends, and
  position-only callbacks differentiated into panel derivatives.
- `tests/geometry/test_near.py`: migrated near-geometry behavior for
  node-distance panel flags, padded rectangular panel flags, and meshgrid
  ordering for rectangular near queries.
- `tests/kernels/test_laplace.py`: Laplace tensor output and first
  singularity metadata checks.
- `tests/kernels/test_helmholtz_singularity.py`: Helmholtz `J0(k*rho) * G`
  singular splitting and finite smooth-remainder checks for scalar,
  derivative, normal-derivative, and hypersingular selectors.
- `tests/kernels/test_kernel_algebra.py`: singular metadata propagation for
  scaling, sums, exact cancellation, and callable Helmholtz coefficients.
- `tests/kernels/test_biharmonic_singularity.py`: biharmonic value, gradient,
  first normal derivative, and Hessian singular metadata against exact
  Laplace-basis product-rule expansions.
- `tests/kernels/test_stokes_singularity.py`: Stokes velocity single-layer
  finite smooth remainder and double-layer exact Laplace-basis singular split.
- `tests/kernels/test_elasticity_singularity.py`: elasticity single-displacement
  exact Laplace-basis singular split.
- `tests/kernels/test_formula_parity.py`: migrated closed-form and
  finite-difference kernel formula checks for Laplace, Helmholtz, biharmonic,
  Stokes, and elasticity kernels through the new kernel API.
- `tests/quadrature/test_panel.py`: dense panel weighting at the quadrature
  boundary, component-major dense operator matrix materialization, and
  adaptive close-panel integration against an analytic straight-panel
  logarithmic integral.
- `tests/quadrature/test_helsing_ojala.py`: Helsing-Ojala log, Cauchy, and
  derivative product weights against oversampled Legendre moments, plus a
  Laplace single-layer close-panel matrix, Laplace double-layer close-panel
  matrix, and Helmholtz log-basis smooth-amplitude dispatch.
- `tests/quadrature/test_ggq.py`: generated GGQ-style self split rules,
  interpolation matrices, and a Laplace single-layer self-panel matrix against
  an analytic straight-segment logarithmic integral.
- `tests/quadrature/test_legendre.py`: migrated Legendre utility behavior for
  coefficient/value transforms, polynomial derivatives, interpolation,
  integration, barycentric weights, Bernstein ellipse points, Taylor stepping,
  and scalar/vector adaptive Gauss integration.
- `tests/rcip/test_primitives.py`: dyadic local corner geometry, barycentric
  prolongation, and scalar/component density interpolation.
- `tests/system/test_density.py`: component-major over panel-major density
  vector adapters.
- `tests/system/test_multi_density_assembly.py`: dense block assembly for
  multiple unknown density spaces and multiple boundary equations, including
  jump insertion and multi-density solve reconstruction.
- `tests/system/test_corrections.py`: component-major dense panel replacement
  insertion for adaptive local quadrature blocks and generated GGQ Laplace
  self-panel replacement.
- `tests/system/test_laplace_exterior_dirichlet.py`: dense exterior Laplace
  double-layer solve, FLAM solve field comparison, plus dense and FMM
  off-boundary evaluation for the unit-circle cosine mode.
- `tests/backends/test_fmm2d.py`: FMM2D Laplace single- and double-layer
  evaluation against dense panel quadrature at off-boundary targets.
- `tests/backends/test_flam.py`: FLAM recursive-skeletonization apply and solve
  compared against a dense reference matrix.
