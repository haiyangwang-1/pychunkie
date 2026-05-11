# Installing fmm2dpy

`pychunkie` depends on `fmm2dpy`, the Python bindings for the Flatiron
Institute FMM2D library. The dependency is pinned to this upstream commit:

```text
550dae5b77b1e006c8ffae37fc832f8c2b536871
```

The upstream repository is <https://github.com/flatironinstitute/fmm2d>. Its
`pyproject.toml` builds the Python package with `scikit-build-core`, CMake, a C
compiler, and a Fortran compiler. Install those native build tools before
running `uv sync`. The same upstream commit is also pinned as the
`external/fmm2d` submodule for CI and local reference setup.

## Verify the install

After following the platform steps below, verify the import:

```bash
uv run python -c "import fmm2dpy; print(fmm2dpy.__file__)"
```

Then run the test suite:

```bash
uv run pytest
```

## Linux

### Ubuntu or Debian

1. Install system build tools:

   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential gfortran cmake git python3-dev
   ```

2. From the `pychunkie` repository root, install dependencies:

   ```bash
   uv sync
   ```

3. Verify the install:

   ```bash
   uv run python -c "import fmm2dpy; print(fmm2dpy.__file__)"
   uv run pytest
   ```

### Fedora, RHEL, or CentOS

1. Install system build tools:

   ```bash
   sudo dnf install -y gcc gcc-c++ gcc-gfortran make cmake git python3-devel
   ```

   On older systems that use `yum`:

   ```bash
   sudo yum install -y gcc gcc-c++ gcc-gfortran make cmake git python3-devel
   ```

2. From the `pychunkie` repository root, install dependencies:

   ```bash
   uv sync
   ```

3. Verify the install:

   ```bash
   uv run python -c "import fmm2dpy; print(fmm2dpy.__file__)"
   uv run pytest
   ```

## macOS

1. Install Apple Command Line Tools:

   ```bash
   xcode-select --install
   ```

2. Install Homebrew from <https://brew.sh>.

3. Install native build tools:

   ```bash
   brew install gcc cmake git
   ```

4. From the `pychunkie` repository root, install dependencies:

   ```bash
   uv sync
   ```

5. Verify the install:

   ```bash
   uv run python -c "import fmm2dpy; print(fmm2dpy.__file__)"
   uv run pytest
   ```

If CMake cannot find Homebrew's GCC/GFortran automatically, point the compiler
environment variables at the Homebrew tools before running `uv sync`:

```bash
export CC="$(brew --prefix gcc)/bin/gcc-15"
export CXX="$(brew --prefix gcc)/bin/g++-15"
export FC="$(brew --prefix gcc)/bin/gfortran"
uv sync
```

Adjust `gcc-15` and `g++-15` if Homebrew installs a newer GCC major version.

## Windows

Use 64-bit Python. The upstream FMM2D build needs a C compiler, a C++ compiler,
CMake, Git, and a Fortran compiler. Microsoft Visual Studio alone is not enough
unless a Fortran compiler is also installed and visible to CMake.

1. Install MSYS2 from <https://www.msys2.org>.

2. Open an MSYS2 UCRT64 shell.

3. Update MSYS2 packages:

   ```bash
   pacman -Syu
   ```

   If MSYS2 asks you to close and reopen the shell, do that, then continue.

4. Install the GNU build toolchain:

   ```bash
   pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gcc-fortran mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-make git
   ```

5. Open PowerShell from the `pychunkie` repository root.

6. Add MSYS2 UCRT64 tools to the current PowerShell session:

   ```powershell
   $env:Path = "C:\msys64\ucrt64\bin;$env:Path"
   ```

7. Tell CMake to use the MSYS2 compilers:

   ```powershell
   $env:CC = "C:\msys64\ucrt64\bin\gcc.exe"
   $env:CXX = "C:\msys64\ucrt64\bin\g++.exe"
   $env:FC = "C:\msys64\ucrt64\bin\gfortran.exe"
   ```

8. Install dependencies:

   ```powershell
   uv sync
   ```

9. Verify the install:

   ```powershell
   uv run python -c "import fmm2dpy; print(fmm2dpy.__file__)"
   uv run pytest
   ```

If the build log says `No CMAKE_Fortran_COMPILER could be found`, CMake still
cannot see `gfortran`. Re-check that `C:\msys64\ucrt64\bin` is on `PATH` and
that `$env:FC` points to `C:\msys64\ucrt64\bin\gfortran.exe`.

## Troubleshooting

### CMake is missing

Install CMake through your OS package manager, or through Python:

```bash
python -m pip install "cmake>=3.19"
```

### Fortran compiler is missing

Install `gfortran`:

- Ubuntu/Debian: `sudo apt-get install gfortran`
- Fedora/RHEL/CentOS: `sudo dnf install gcc-gfortran`
- macOS: `brew install gcc`
- Windows: install MSYS2 and `mingw-w64-ucrt-x86_64-gcc-fortran`

### Runtime loading fails after import

Make sure compiler runtime libraries are discoverable:

- Linux: confirm the relevant compiler runtime directory is on
  `LD_LIBRARY_PATH` if needed.
- macOS: confirm the relevant compiler runtime directory is on
  `DYLD_LIBRARY_PATH` if needed.
- Windows: confirm `C:\msys64\ucrt64\bin` is on `PATH`.
