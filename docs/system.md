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
- Dense self assembly inserts the finite smooth-boundary diagonal limit for
  Laplace double-layer and adjoint double-layer traces, so Dirichlet and
  Neumann circle solves have finite reference matrices before special
  quadrature owns all same-panel corrections.
- Right-hand sides may be scalar panel data, component panel data, or already
  flattened component-major vectors, and multi-equation right-hand sides are
  concatenated in equation order.
- `Constraint` and `ConstraintTerm` append explicit dense rows after boundary
  equations for compatibility conditions such as scalar charge constraints.
- `PanelCorrection`, `build_panel_correction`, and `build_corrections` provide
  dense replacement-block insertion and first-pass automatic self/near panel
  selection for adaptive, Helsing-Ojala, and GGQ local panel matrices.
  Generated GGQ self-panel replacement is active for Laplace single-layer
  blocks.
- `build_rcip_state` discovers nonsmooth `ChunkGraph` vertices, builds dyadic
  local corner geometry plus split-panel prolongation blocks, and records RCIP
  state in `SystemMatrix.diagnostics`. Eligible scalar second-kind graph
  systems use the old RCIP star-block replacement during dense assembly, and
  field evaluation uses the old coarse-density plus reconstructed local-corner
  contribution split.
- `LaplaceExteriorDirichletSystem` solves the unit-circle cosine mode and
  evaluates the exterior field at an off-boundary target.
- The smooth Laplace examples are standalone scripts that build the same
  system records directly for interior/exterior Dirichlet and Neumann circle
  problems, then write solution and `log10(abs(error))` field figures.
- The nonsmooth square examples are standalone scripts that build `ChunkGraph`
  boundary systems for the same four Laplace cases, use the same RCIP solve and
  evaluation behavior as the old examples, and write matching solution/error
  figures. Their field plots use dense evaluation with adaptive close-panel
  replacement so targets close to a side do not use ordinary panel Gauss alone.
- Dense and GMRES solves reconstruct one `Density` object per unknown block.
  The first GMRES path uses SciPy's `LinearOperator` over the dense reference
  matrix and records solver diagnostics on `SystemSolution`.
- `SystemConfig(solve_method="flam")` routes scalar dense-reference solves
  through the FLAM backend before reconstructing density objects. Multiple
  scalar unknown blocks are supported by adding a backend-only block coordinate
  so repeated geometry points remain distinct for pyFLAM.
- Dense field evaluation honors `SystemConfig.close_correction` by replacing
  close source-panel contributions with adaptive local panel matrices.
- Field evaluation honors `SystemConfig.evaluation_method="fmm"` for supported
  Laplace layer potentials and compares against dense evaluation. FMM
  evaluation remains an off-boundary backend path and does not apply adaptive
  close-panel replacement.
- `matrix_free_matvec` and `SystemOperator` apply trace terms directly through
  point-view kernel contractions without materializing the global dense matrix.
- `fmm_matvec` applies supported scalar Laplace trace terms with FMM2D for
  off-boundary systems and compares against dense assembly.
- Broader accelerated matvecs, vector/multi-density RCIP, and non-straight
  graph-edge RCIP parity remain required upcoming work.
- `docs/structured-rskelf-transmission.md` records the transmission-system and
  structured RSKELF design target for multiple boundaries and densities.
