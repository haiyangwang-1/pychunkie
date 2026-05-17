# Python Test Suite Summary

Status: reset for the clean rewrite.

Update trigger: update this file when tests are added, removed, renamed,
parametrized, or when the behavior covered by existing tests changes
meaningfully.

## Current Snapshot

- Previous test-suite summary is archived in
  `docs_old/docs/python-test-suite-summary.md`.
- Active tests currently cover the rewrite bootstrap and first low-level APIs.
- Verification snapshot: `uv run pytest -q` on 2026-05-16,
  `13 passed` in 0.96 seconds.

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
  point maps, polygon construction, and near-panel flags.
- `tests/kernels/test_laplace.py`: Laplace tensor output and first
  singularity metadata checks.
- `tests/kernels/test_helmholtz_singularity.py`: Helmholtz `J0(k*rho) * G`
  singular splitting and finite smooth-remainder checks for scalar,
  derivative, normal-derivative, and hypersingular selectors.
- `tests/quadrature/test_panel.py`: dense panel weighting at the quadrature
  boundary.
- `tests/system/test_density.py`: component-major over panel-major density
  vector adapters.
- `tests/system/test_laplace_exterior_dirichlet.py`: dense exterior Laplace
  double-layer solve and off-boundary evaluation for the unit-circle cosine
  mode.
