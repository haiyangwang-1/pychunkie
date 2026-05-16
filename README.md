# pychunkie

`pychunkie` is being rewritten as a Python-first boundary integral equation
toolkit. The previous implementation, tests, and historical docs are archived
under `src_old/`, `tests_old/`, and `docs_old/` for reference.

The active implementation follows `design.md`:

- `chunkie.geometry` owns boundary geometry and point storage.
- `chunkie.kernels` owns PDE kernel formulas and singularity metadata.
- `chunkie.quadrature` owns local panel rules.
- `chunkie.rcip` owns corner compression primitives.
- `chunkie.system` owns assembly, traces, solves, and evaluation.

Use `uv` for the package environment:

```powershell
uv run pytest
uv run ruff check .
uv run mypy
```

The rewrite is intentionally clean. Old MATLAB-shaped public names are reference
material, not compatibility contracts.
