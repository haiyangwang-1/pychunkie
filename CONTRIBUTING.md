# Contributing

This repository is Python-first. MATLAB chunkIE is an important numerical
reference and fixture source, but it is not the public API contract for this
package. Prefer readable Python interfaces, ordinary NumPy semantics, and code
that makes the underlying numerical formula visible.

## Design Philosophy

- Preserve mathematical structure in code. Prefer clear tensor operations,
  named axes, and explicit adapter boundaries over layout tricks.
- Keep dense/direct implementations as the reference path for correctness.
  Accelerated implementations must be checked against dense/direct behavior.
- Treat MATLAB parity as numerical parity where relevant, not as a requirement
  to preserve MATLAB names, flattening conventions, or interleaved layouts.
- Put compatibility complexity at the boundary to external solvers, FMM/FLAM,
  MATLAB fixtures, sparse matrices, and legacy entry points.
- Avoid broad refactors unless they remove real ambiguity, reduce meaningful
  risk, or are done as a dedicated mechanical cleanup.

## API, Naming, And Style

Python-facing APIs and new internal code should use clear domain names. Existing
MATLAB-style public APIs may be renamed or removed when they obstruct clarity;
this repo is new enough that compatibility aliases are optional.

Use these names in new and touched code unless a smaller local formula is more
readable:

- `quadrature_order` instead of `k`
- `chunker` instead of `chnkr`
- `kernel` instead of `kern`
- `options` instead of `opts`
- `source` and `target` instead of `src` and `targ` outside tiny scopes
- `positions`, `derivatives`, `second_derivatives`, `normals`, and `weights`
  when a field name would otherwise be unexplained

Short local names are acceptable when they are standard mathematical notation or
obvious inside a small formula. Established specialized algorithm names such as
`ggq`, `rcip`, and `flam` are domain vocabulary and should remain concise.

FLAM callback signatures may need shorthand imposed by external APIs. Wrap those
callbacks or add comments at the callback boundary so the meaning of arguments
such as self indices, neighbor indices, box size, center, proxy nodes, and proxy
weights is clear.

## Shape And Tensor Conventions

Treat 1D arrays as ordinary vectors, 2D arrays as ordinary matrices, and arrays
with three or more dimensions as tensors with meaningful axes. Do not reshape or
flatten just to mimic MATLAB storage when the computation can be expressed on the
natural tensor shape.

Use these axis symbols in `np.einsum` strings and nearby documentation:

- `R`: coordinate axis
- `s`: source node
- `S`: source chunk
- `t`: target node
- `T`: target chunk
- `d`: density or input component
- `f`: field or output component

Canonical shapes:

- `source_positions[R, s, S]`
- `source_derivatives[R, s, S]`
- `source_normals[R, s, S]`
- `source_weights[s, S]`
- `density[d, s, S]`
- `field[f, t, T]`
- `kernel_values[f, t, T, d, s, S]`

Prefer tensor operations and `np.einsum` for contractions, component mixing,
kernel assembly, and quadrature application. For example:

```python
field = np.einsum("ftTdsS,dsS,sS->ftT", kernel_values, density, source_weights)
```

Keep explicit loops and indexing for scalar recurrences, graph traversal, Newton
iterations, adaptive subdivision, and other genuinely sequential algorithms.

Flat vectors, interleaved component layouts, and `reshape(..., order="F")` are
adapter formats only. Use them at explicit boundaries such as `np.linalg`,
`scipy.sparse`, `LinearOperator`, `fmm2dpy`, `pyflam`, MATLAB fixture loading, or
temporary legacy API conversion. Name or comment those conversions so their
boundary role is clear.

## Numerical Tests And Performance

Numerical tests should use pointwise absolute-or-relative acceptance: each point
passes when either `abs_err <= abs_tol` or `rel_err <= rel_tol`. Do not rely on a
plain relative tolerance when the reference value may be close to zero.

Tests should still log both maximum absolute error and maximum relative error.
When a test exercises acceleration, also record useful context such as problem
size, backend, elapsed time, tolerance, and any speed ratio being compared.

Routine correctness tests should use generous timeouts to catch hangs. Opt-in
performance tests should be marked and should not impose hard speed-ratio gates
unless the environment is controlled enough for the threshold to be meaningful.

Use pytest markers consistently:

- `slow`: expensive but ordinary correctness coverage
- `performance`: opt-in timing or scaling checks
- `requires_matlab`: requires a local MATLAB executable or MATLAB fixture refresh

## Tooling

Use `uv` for the package-level Python environment. Prefer these commands:

```powershell
uv run pytest
uv run ruff format .
uv run ruff check .
uv run mypy
```

Formatting or lint-only sweeps should be their own mechanical changes. Do not
hide formatting churn inside feature, parity, or test-strengthening commits.

## Living Docs And Commits

All project-owned Markdown docs are living docs. A doc should either have a
clear owner and update trigger, or it should be deleted when it becomes
obsolete. Markdown files inside `external/` are upstream/reference material and
are updated only through the corresponding vendored checkout or submodule.

Before committing code, tests, fixtures, or docs, review whether `map.md` needs
an update. In particular, follow the status-document rules in `AGENTS.md`.

Keep commits small and logically complete. A behavior change should include the
relevant implementation, tests, fixture updates, and living-doc updates in the
same commit. If no living-doc update is needed, say why in the commit message or
PR summary.
