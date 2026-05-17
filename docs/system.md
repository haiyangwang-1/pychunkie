# System

`chunkie.system` owns high-level boundary integral equation workflows:

- density storage and solver-vector layout,
- layer potentials,
- traces and jump terms,
- equations and constraints,
- block layout,
- matrix assembly,
- corrections,
- solves,
- and field evaluation.

Jump terms are trace metadata, not hidden kernel behavior. The first convenience
system is the Laplace exterior Dirichlet formulation:

```text
(+0.5 I + D) sigma = boundary_data
u(x) = D[sigma](x)
```

Current implementation status:

- Density layout and system records are active.
- Dense assembly supports multiple unknown density blocks and multiple boundary
  equations for `Chunker` and `ChunkGraph` `BoundaryPart` trace terms, with
  trace blocks materialized through the component-major operator matrix layout.
- Right-hand sides may be scalar panel data, component panel data, or already
  flattened component-major vectors, and multi-equation right-hand sides are
  concatenated in equation order.
- `PanelCorrection`, `build_panel_correction`, and `build_corrections` provide
  dense replacement-block insertion and first-pass automatic self/near panel
  selection for adaptive, Helsing-Ojala, and GGQ local panel matrices.
  Generated GGQ self-panel replacement is active for Laplace single-layer
  blocks.
- `build_rcip_state` discovers nonsmooth `ChunkGraph` vertices, builds dyadic
  local corner geometry plus split-panel prolongation blocks, and dense
  assembly records active RCIP state in `SystemMatrix.diagnostics` without
  changing the dense reference matrix.
- `LaplaceExteriorDirichletSystem` solves the unit-circle cosine mode and
  evaluates the exterior field at an off-boundary target.
- Dense solves reconstruct one `Density` object per unknown block.
- `SystemConfig(solve_method="flam")` routes scalar one-unknown dense-reference solves
  through the FLAM backend before reconstructing the density object.
- Field evaluation honors `SystemConfig.evaluation_method="fmm"` for supported
  Laplace layer potentials and compares against dense evaluation.
- `fmm_matvec` applies supported scalar Laplace trace terms with FMM2D for
  off-boundary systems and compares against dense assembly.
- Constraints, recursive RCIP Schur updates, nonsmooth correction insertion,
  and broader accelerated matvecs remain required upcoming work.
- `docs/structured-rskelf-transmission.md` records the transmission-system and
  structured RSKELF design target for multiple boundaries and densities.
