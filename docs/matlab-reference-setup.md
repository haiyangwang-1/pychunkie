# MATLAB Reference Setup

Initialize the pinned test/parity reference dependencies from the repository
root:

```powershell
git submodule update --init --recursive
```

Current test-only submodule pins:

- `external/chunkie-matlab`: `af34cc41c81114e693b515066e4d308067bf7e63`
  for MATLAB fixture generation and parity inspection.
- `external/FLAM`: `73b7accda7c1a933517b008831d8404d8d3cc764`
  for MATLAB-side FLAM parity references.
- `external/fmm2d`: `550dae5b77b1e006c8ffae37fc832f8c2b536871`
  for MATLAB-side FMM2D reference and MEX parity setup.

These submodules are not Python package runtime or build dependencies.

Local status:

- MATLAB R2026a is callable via `matlab -batch`.
- `external/chunkie-matlab/startup.m` runs far enough for dense MATLAB
  fixture generation.
- FMM2D MATLAB MEX builds and loads on this Windows machine when MATLAB is
  configured to use MSYS2 MinGW for C MEX compilation.

Current FMM2D status:

- MSYS2 was installed through `winget`.
- `mingw-w64-x86_64-gcc`, `mingw-w64-x86_64-gcc-fortran`, and `make`
  were installed through MSYS2 `pacman`.
- The FMM2D Fortran static library builds with MSYS2 GFortran.
- `fmm2d.mexw64` builds when MATLAB's C MEX compiler is explicitly set to
  `mingw64.xml` and `MW_MINGW64_LOC=C:\msys64\mingw64` is visible to
  `mex.bat`.
- `scripts/matlab/fixture_context.m` prepends `C:\msys64\mingw64\bin` to
  MATLAB's `PATH` so the MEX can load MinGW runtime DLLs during fixture
  generation.

To rebuild the Windows MATLAB MEX from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\matlab\build_fmm2d_mex_windows.ps1
```

The manual equivalent is:

```powershell
$env:MW_MINGW64_LOC = "C:\msys64\mingw64"
matlab -batch "setenv('MW_MINGW64_LOC','C:\msys64\mingw64'); mex -setup:'C:\Program Files\MATLAB\R2026a\bin\win64\mexopts\mingw64.xml' C"
C:\msys64\usr\bin\bash.exe -lc "cd /c/Users/haiya/git/pychunkie/external/chunkie-matlab/chunkie/fmm2d && export MW_MINGW64_LOC=/mingw64 && export PATH=/mingw64/bin:/usr/bin:`$PATH && make -B matlab OMP=OFF"
```

Notes:

- Use `make -B matlab`; upstream's `matlab` target name collides with the
  existing `matlab/` directory, so plain `make matlab` can incorrectly report
  "up to date" before a MEX exists.
- Run through MSYS2 `bash`; the upstream Makefile uses Unix commands such as
  `mv`.
- Configure MATLAB MEX for MinGW explicitly. If MATLAB uses MSVC, it either
  rejects C99 `_Complex` in `matlab/fmm2d.c` or searches for MSVC-style
  `gfortran.lib` instead of MinGW libraries.
- The build emits many upstream Fortran rank/type warnings. The successful
  signal is `MEX completed successfully` and the existence of
  `external/chunkie-matlab/chunkie/fmm2d/matlab/fmm2d.mexw64`.

The Python port declares the upstream `fmm2dpy` Python package as a runtime
dependency in `pyproject.toml`, with its pinned FMM2D source recorded under
`[tool.uv.sources]`. That is separate from MATLAB MEX support: the MEX build
above is only needed for running MATLAB-side FMM2D reference code, not for
importing `chunkie` in Python.
