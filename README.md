# pychunkie

A Python port of the MATLAB [chunkIE](https://github.com/fastalgorithms/chunkie)
package.

The port is intentionally conservative:

- mirror the MATLAB package structure where practical under `src/chunkie`;
- use `uv` and the project-local `.venv`;
- keep runtime dependencies to `numpy` and `scipy` for the first milestones;
- defer `fmm2dpy` and FLAM until dense/direct functionality is correct;
- test behavior against MATLAB-generated golden fixtures as the port grows.

## Development

```powershell
uv sync
uv run pytest
```

MATLAB reference checkouts live in ignored `external/` directories for local
inspection and fixture generation. The Python package does not import them.
