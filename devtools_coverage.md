# Devtools Coverage

Status: reset for the clean rewrite.

Update trigger: update this file when MATLAB devtools parity coverage changes,
including golden fixtures, MATLAB fixture generators, Python fixture generators,
parity tests, ranked port status, or suggested next ports.

## Current Snapshot

- Previous devtools coverage notes are archived in `docs_old/devtools_coverage.md`.
- The old Python-side devtools snapshot generator is archived in
  `docs_old/scripts/`; it depends on removed public APIs.
- No active rewrite devtools parity tests have been migrated yet.
- Current active verification is `uv run pytest -q`, `23 passed` in
  1.57 seconds, on 2026-05-16.
- GGQ, RCIP, `ChunkGraph`, FMM, and FLAM parity remain required milestones.

## Rewrite Policy

Old tests and fixtures should be migrated by behavior, not by preserving old
public API names. Keep MATLAB wrappers under `scripts/matlab`, compact fixture
data under `tests/golden`, and Python comparisons under `tests`.

Do not edit `external/chunkie-matlab/devtools/test` for parity work.
