# pychunkie

A Python port of the MATLAB [chunkIE](https://github.com/fastalgorithms/chunkie)
package.

The port is intentionally conservative:

- mirror the MATLAB package structure where practical under `src/chunkie`;
- use `uv` and the project-local `.venv`;
- keep runtime dependencies focused on `numpy`, `scipy`, `fmm2dpy`, and
  `matplotlib` for demo figures;
- add FMM2D acceleration incrementally while dense/direct functionality remains
  the reference path;
- test behavior against MATLAB-generated golden fixtures as the port grows.

## Development

```powershell
git submodule update --init --recursive
uv sync
uv run pytest
uv run python scripts/clean_test_data.py
```

See [docs/fmm2dpy-install.md](docs/fmm2dpy-install.md) for macOS, Windows, and
Linux notes on installing the upstream `fmm2dpy` dependency.

MATLAB and native-library reference checkouts live in pinned, test-time-only
`external/` submodules for local inspection, fixture generation, and CI setup.
The Python package does not import them.
MATLAB parity fixtures under `tests/golden` are generated on demand by tests,
are ignored by Git, and require `external/chunkie-matlab` when regenerated.
The runtime `fmm2dpy` package is pinned separately in `pyproject.toml` through
uv's source table and built from the upstream FMM2D repository during install.

## Documentation And Demos

- [docs/bie-overview.md](docs/bie-overview.md) describes the core BIE API,
  kernel families, chunkgraphs, and dense/FMM/FLAM workflows.
- `examples/smooth_laplace_*_{dirichlet,neumann}.py` solve one smooth Laplace
  interior or exterior model problem per file.
- `examples/nonsmooth_laplace_*_{dirichlet,neumann}.py` solve one square BVP
  per file and write corrected near-boundary solution/error PNGs next to the
  script.
- `examples/chunkgraph_region_classification.py` and
  `examples/chunkgraph_annular_dirichlet.py` demonstrate chunkgraph regions and
  a multiply connected BVP.
- `examples/accelerated_fmm_*.py` and `examples/accelerated_flam_laplace.py`
  show one accelerated physics/backend combination per file.
