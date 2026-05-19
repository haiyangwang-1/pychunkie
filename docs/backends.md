# Backends

`chunkie.system.backends` is a placeholder namespace. The active rewrite does
not currently support `fmm2dpy` or `pyflam` acceleration, and neither package is
a runtime dependency.

Deferred backend milestones:

- FMM-backed matvec and evaluation compared against dense references.
- FLAM apply and solve compared against dense references.

Backends consume kernel metadata and system layout. Kernel objects do not call
FMM or FLAM directly.

Current implementation status:

- `SystemConfig` accepts dense evaluation and dense or GMRES solves. It rejects
  `evaluation_method="fmm"` and `solve_method="flam"`.
- `system.matrix_free_matvec` remains a dense-reference matrix-free path. There
  is no supported FMM-backed matvec path.
- FMM2D and FLAM files under `external/` remain upstream reference/support
  checkouts only; package code does not import `fmm2dpy` or `pyflam`.
- `docs/structured-rskelf-transmission.md` defines the structured RSKELF target:
  pyFLAM should consume row/column density metadata and per-block proxy samples
  instead of receiving an already-flattened transmission matrix.
