# pychunkie

A Python-first boundary-integral toolkit inspired by the MATLAB
[chunkIE](https://github.com/fastalgorithms/chunkie) package.

The MATLAB codebase remains the numerical reference and fixture source, but it
is not the public API contract for this package. The repo now favors ordinary
NumPy semantics, clear domain names, and explicit adapter boundaries for
solver/backend layouts.

- keep MATLAB numerical parity where relevant while using Python-centered
  package boundaries and APIs under `src/chunkie`;
- use `uv` and the project-local `.venv`;
- keep runtime dependencies focused on `numpy`, `scipy`, `fmm2dpy`, `pyflam`,
  and `matplotlib` for demo figures;
- keep dense/direct functionality as the correctness reference for FMM/FLAM
  acceleration;
- test behavior against MATLAB-generated golden fixtures as the port grows.

## Installation

From the repository root, install the package with `uv`:

```powershell
uv sync
```

Then run an example in the managed environment:

```powershell
uv run python examples/smooth_laplace_interior_dirichlet.py
```

Without `uv`, use a standard virtual environment and `pip`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Then run the same example with that environment's Python:

```powershell
python examples/smooth_laplace_interior_dirichlet.py
```

`pip` installs need Git available because `fmm2dpy` and `pyflam` are pinned
from Git repositories in `pyproject.toml`.

## Development

```powershell
git submodule update --init --recursive
uv sync
uv run pytest
uv run ruff check .
uv run mypy
uv run python scripts/clean_test_data.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the Python-first design philosophy,
tensor notation, naming standard, testing policy, and living-doc rules.

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

The examples intentionally use Python-first variable names and keyword options
instead of MATLAB-layout dictionary keys.
