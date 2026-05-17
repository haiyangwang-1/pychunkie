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
  `27 passed` in 1.96 seconds; `uv run ruff check .` passes; `uv run mypy`
  passes over `src/chunkie`.

## Source Tree

| Area | Status | Notes |
| --- | --- | --- |
| `chunkie.geometry` | in progress | Panel-major `Chunker`, point maps, basic constructors, and `ChunkGraph` data records are the first active layer. |
| `chunkie.kernels` | in progress | Kernel object, Laplace/Helmholtz/biharmonic/Stokes/elasticity formulas, registry, algebra, and singularity metadata foundation are active. Laplace metadata uses `G`, `G_a`, and `G_ab`; smooth amplitudes are allowed on those bases for special-quadrature consumption; Helmholtz metadata differentiates `J0(k*rho) * G`; biharmonic metadata uses `B=-(rho^2/4)G`; Stokes velocity metadata covers `s` and `d`; elasticity single-displacement metadata covers `s`; algebra scales and cancels exact scalar/matrix singular terms. |
| `chunkie.quadrature` | in progress | Dense panel helpers, component-major dense operator materialization, adaptive source-panel fallback, Helsing-Ojala log-product weights, and log-basis `SingularityInfo` smooth-amplitude dispatch are active; GGQ parity and derivative-basis singularity dispatch are required upcoming milestones. |
| `chunkie.rcip` | scaffolded | RCIP is required for the rewrite and is not treated as deferred. |
| `chunkie.system` | in progress | Density layout, dense one-unknown/one-equation assembly, solve, evaluation, and `LaplaceExteriorDirichletSystem` are active for the first Laplace circle regression. Trace blocks use the quadrature operator-matrix adapter. |
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
