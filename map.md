# Rewrite Implementation Map

Status: active clean rewrite.

Update trigger: update this file when public APIs, source-tree structure,
implemented behavior, known limitations, or verification snapshots change.

## Current Snapshot

- Source rewrite bootstrapped from `design.md`.
- Previous implementation archived in `src_old/`.
- Previous tests archived in `tests_old/`.
- Previous docs and status notes archived in `docs_old/`.
- Verification snapshot: `uv run pytest -q` on 2026-05-17,
  `86 passed` in 3.50 seconds; `uv run ruff check .` passes; `uv run mypy`
  passes over `src/chunkie`.

## Source Tree

| Area | Status | Notes |
| --- | --- | --- |
| `chunkie.geometry` | in progress | Panel-major `Chunker`, point maps, basic constructors, metrics, arclength parameterization/resampling, uniform split-panel refinement, affine/rotate/reflect transforms, near-panel node-distance flags, rectangle flags, nearest-point projection, and operational single- and multi-edge `ChunkGraph` records/views/classification are active. |
| `chunkie.kernels` | in progress | Kernel object, Laplace/Helmholtz/biharmonic/Stokes/elasticity formulas, registry, algebra, and singularity metadata foundation are active. Laplace metadata uses `G`, `G_a`, and `G_ab`; smooth amplitudes are allowed on those bases for special-quadrature consumption; Helmholtz metadata differentiates `J0(k*rho) * G`; biharmonic metadata uses `B=-(rho^2/4)G`; Stokes velocity metadata covers `s` and `d`; elasticity single-displacement metadata covers `s`; algebra scales and cancels exact scalar/matrix singular terms. |
| `chunkie.quadrature` | in progress | Legendre polynomial/transform/interpolation/integration utilities, dense panel helpers, component-major dense operator materialization, adaptive source-panel fallback, Helsing-Ojala log/Cauchy product weights, generated GGQ-style split rules/self-panel matrices, fixture-backed GGQ removable-rule parity, log-basis `SingularityInfo` smooth-amplitude dispatch, and Helmholtz single-layer HO panel correction are active; broader MATLAB GGQ table parity and derivative-basis singularity dispatch remain required milestones. |
| `chunkie.rcip` | in progress | Dyadic local corner geometry, barycentric and split-panel prolongation matrices, edge/component block prolongation, dense Schur compression updates, and density interpolation are active. Recursive drivers and system integration remain required upcoming work. |
| `chunkie.system` | in progress | Density layout, dense multi-unknown/multi-equation `Chunker` and `BoundaryPart` trace assembly, dense multi-density solve reconstruction, dense/FMM evaluation and scalar Laplace FMM matvec, adaptive/Helsing-Ojala/GGQ panel replacement corrections, and `LaplaceExteriorDirichletSystem` are active. Trace blocks use the quadrature operator-matrix adapter. FLAM solve remains scalar one-unknown only. |
| `chunkie.system.backends` | in progress | FMM2D evaluates scalar Laplace single- and double-layer potentials and drives scalar off-boundary system matvecs against dense references. FLAM factors dense reference matrices with `pyflam.rskelf` and tests apply/solve against dense linear algebra. `docs/structured-rskelf-transmission.md` records the structured RSKELF target for block transmission systems. |

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
- Structured RSKELF/pyFLAM callbacks for multi-boundary, multi-density systems.
- FMM-backed matvec/evaluation.
- FLAM apply/solve backend.
- Exact singularity expansion metadata with smooth-remainder tests.
