# MATLAB Reference Setup

Local status:

- MATLAB R2026a is callable via `matlab -batch`.
- `external/chunkie-matlab/startup.m` runs far enough for dense MATLAB
  fixture generation.
- FMM2D MATLAB MEX is not built yet on this Windows machine.

Current FMM2D blocker:

- MATLAB detects Microsoft Visual C++ 2022 for C MEX compilation.
- MATLAB does not detect a supported Fortran compiler.
- `gcc` and `gfortran` are not currently on `PATH`.

The upstream FMM2D Windows instructions expect MinGW/GNU tooling:

1. Install GNU make, GCC, and GFortran/MinGW.
2. Copy `make.inc.windows.mingw` to `make.inc` inside
   `external/chunkie-matlab/chunkie/fmm2d`.
3. Update `MINGW_LPATH` in `make.inc` if needed.
4. Run `make matlab` from `external/chunkie-matlab/chunkie/fmm2d`.
5. Re-run `external/chunkie-matlab/startup.m` and upstream MATLAB tests.

The Python port currently does not depend on FMM2D or FLAM.
