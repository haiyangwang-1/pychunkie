# MATLAB Reference Setup

Local status:

- MATLAB R2026a is callable via `matlab -batch`.
- `external/chunkie-matlab/startup.m` runs far enough for dense MATLAB
  fixture generation.
- FMM2D MATLAB MEX is not built yet on this Windows machine.

Current FMM2D status:

- MSYS2 was installed through `winget`.
- `mingw-w64-x86_64-gcc`, `mingw-w64-x86_64-gcc-fortran`, and `make`
  were installed through MSYS2 `pacman`.
- The FMM2D Fortran static library builds with MSYS2 GFortran.
- The final MATLAB MEX link still fails because MATLAB MEX is configured for
  Microsoft Visual C++ 2022 and does not accept the MinGW `libgfortran.a`
  import library through `-lgfortran`.

The remaining FMM2D blocker is MATLAB-side MinGW MEX configuration. The
upstream FMM2D Windows instructions expect MinGW/GNU tooling:

1. Install/configure MATLAB's MinGW-w64 MEX support.
2. Copy `make.inc.windows.mingw` to `make.inc` inside
   `external/chunkie-matlab/chunkie/fmm2d`.
3. Update `MINGW_LPATH`, `FC`, `CC`, `CXX`, and `MEX` in `make.inc` if needed.
4. Run `make matlab` from `external/chunkie-matlab/chunkie/fmm2d`.
5. Re-run `external/chunkie-matlab/startup.m` and upstream MATLAB tests.

The Python port currently does not depend on FMM2D or FLAM.
