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
- Scalar dense assembly supports one unknown and one boundary equation.
- `LaplaceExteriorDirichletSystem` solves the unit-circle cosine mode and
  evaluates the exterior field at an off-boundary target.
- General block systems, constraints, nonsmooth corrections, and accelerated
  matvecs remain required upcoming work.
