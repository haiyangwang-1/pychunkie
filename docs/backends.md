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
  path through `SystemConfig(evaluation_method="fmm")`.
- `system.backends.flam.factor_system` builds a `pyflam.rskelf` factor from a
  dense reference matrix and exposes apply/solve comparisons against dense
  linear algebra. Callback-based assembly remains upcoming.
