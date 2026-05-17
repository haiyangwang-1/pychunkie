# Devtools Coverage

Status: reset for the clean rewrite.

Update trigger: update this file when MATLAB devtools parity coverage changes,
including golden fixtures, MATLAB fixture generators, Python fixture generators,
parity tests, ranked port status, or suggested next ports.

## Current Snapshot

- Previous devtools coverage notes are archived in `docs_old/devtools_coverage.md`.
- The old Python-side devtools snapshot generator is archived in
  `docs_old/scripts/`; it depends on removed public APIs.
- Active rewrite devtools parity starts with `tests/golden/quadggq.mat` and
  `tests/quadrature/test_ggq.py`, covering generated removable-rule parity
  against archived MATLAB fixture data.
- Current active verification is `uv run pytest -q`, `102 passed` in
  4.31 seconds, on 2026-05-17.
- Broader MATLAB table-backed GGQ parity, RCIP recursive-compression parity,
  and MATLAB-backed `ChunkGraph` parity remain required milestones.
  FMM has active dense-reference tests for scalar Laplace layer evaluation but
  does not yet have MATLAB fixture parity in the rewrite tree.
  FLAM has active dense-reference apply/solve tests but does not yet have
  callback or MATLAB fixture parity in the rewrite tree.
  Helsing-Ojala has active local log, Cauchy, and derivative-product baselines
  but does not yet have MATLAB fixture parity in the rewrite tree.

## Rewrite Policy

Old tests and fixtures should be migrated by behavior, not by preserving old
public API names. Keep MATLAB wrappers under `scripts/matlab`, compact fixture
data under `tests/golden`, and Python comparisons under `tests`.

Do not edit `external/chunkie-matlab/devtools/test` for parity work.
