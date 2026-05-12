# pychunkie

A Python port of the MATLAB [chunkIE](https://github.com/fastalgorithms/chunkie)
package.

The port is intentionally conservative:

- mirror the MATLAB package structure where practical under `src/chunkie`;
- use `uv` and the project-local `.venv`;
- keep runtime dependencies focused on `numpy`, `scipy`, and `fmm2dpy`;
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
