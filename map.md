# Rewrite Implementation Map

Status: active clean rewrite.

Update trigger: update this file when public APIs, source-tree structure,
implemented behavior, known limitations, or verification snapshots change.

## Current Snapshot

- Source rewrite bootstrapped from `design.md`.
- Previous implementation archived in `src_old/`.
- Previous tests archived in `tests_old/`.
- Previous docs and status notes archived in `docs_old/`.
- Verification snapshot: `uv run pytest -q` on 2026-05-16,
  `11 passed` in 0.80 seconds; `uv run ruff check .` passes; `uv run mypy`
  passes over `src/chunkie`.

## Source Tree

| Area | Status | Notes |
| --- | --- | --- |
| `chunkie.geometry` | in progress | Panel-major `Chunker`, point maps, basic constructors, and `ChunkGraph` data records are the first active layer. |
| `chunkie.kernels` | in progress | Kernel object, Laplace formulas, registry, algebra hooks, and singularity metadata foundation are active. The math track recommends Laplace-basis metadata built from `G`, `G_a`, and `G_ab`. |
| `chunkie.quadrature` | scaffolded | Dense panel helpers are first; Helsing-Ojala and GGQ parity are required upcoming milestones. |
| `chunkie.rcip` | scaffolded | RCIP is required for the rewrite and is not treated as deferred. |
| `chunkie.system` | in progress | Density layout, scalar dense assembly, solve, evaluation, and `LaplaceExteriorDirichletSystem` are active for the first Laplace circle regression. |
| `chunkie.system.backends` | scaffolded | FMM and FLAM are required rewrite milestones after dense references are stable. |

## Public API Target

The top-level package should expose stable user-facing objects:

- `Chunker`
- `ChunkGraph`
- `Density`
- `Kernel`
- `kernel`
- `LayerPotential`
- `BoundaryTrace`
- `BoundaryEquation`
- `IntegralSystem`
- `SystemMatrix`
- `SystemSolution`

## Required Milestones

These are required rewrite work, not optional deferrals:

- Helsing-Ojala panel quadrature.
- GGQ fixture-backed parity.
- RCIP local compression and system integration.
- `ChunkGraph` multi-region systems.
- FMM-backed matvec/evaluation.
- FLAM apply/solve backend.
- Exact singularity expansion metadata with smooth-remainder tests.
