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
- `PanelCorrection` and `build_panel_correction` provide the first dense
  replacement-block insertion boundary for adaptive and Helsing-Ojala local
  panel matrices. Generated GGQ self-panel replacement is active for Laplace
  single-layer blocks.
- `LaplaceExteriorDirichletSystem` solves the unit-circle cosine mode and
  evaluates the exterior field at an off-boundary target.
- Dense solves reconstruct one `Density` object per unknown block.
- `SystemConfig(solve_method="flam")` routes scalar one-unknown dense-reference solves
  through the FLAM backend before reconstructing the density object.
- Field evaluation honors `SystemConfig.evaluation_method="fmm"` for supported
  Laplace layer potentials and compares against dense evaluation.
- General `ChunkGraph` systems, constraints, nonsmooth corrections, and
  accelerated matvecs remain required upcoming work.
- `docs/structured-rskelf-transmission.md` records the transmission-system and
  structured RSKELF design target for multiple boundaries and densities.
