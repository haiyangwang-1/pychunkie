# Backends

Acceleration lives under `chunkie.system.backends`.

Required backend milestones:

- FMM-backed matvec and evaluation compared against dense references.
- FLAM apply and solve compared against dense references.

Backends consume kernel metadata and system layout. Kernel objects do not call
FMM or FLAM directly.

Current implementation status:

- `system.backends.fmm2d.apply_fmm` evaluates scalar Laplace single- and
  double-layer potentials with `fmm2dpy` and compares against dense panel
  quadrature at off-boundary targets. `SystemSolution.evaluate` can use this
  path through `SystemConfig(evaluation_method="fmm")`; `system.fmm_matvec`
  uses the same backend for scalar off-boundary Laplace trace matvecs.
- `system.backends.flam.factor_system` builds a `pyflam.rskelf` factor from a
  dense reference matrix and exposes apply, solve, and log-determinant
  comparisons against dense linear algebra, including complex multiple-RHS
  inputs. `SystemConfig(solve_method="flam")` can use this path for scalar
  dense-reference systems with one or more unknown blocks; repeated geometry
  points are separated by a backend-only block coordinate before factorization.
  Callback-based assembly remains upcoming.
- `docs/structured-rskelf-transmission.md` defines the structured RSKELF target:
  pyFLAM should consume row/column density metadata and per-block proxy samples
  instead of receiving an already-flattened transmission matrix.
