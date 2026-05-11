# Golden Fixtures

This directory stores compact MATLAB-generated fixtures used for parity tests.

Generate them after MATLAB chunkIE is working locally:

```powershell
matlab -batch "run('scripts/matlab/generate_basic_fixtures.m')"
```

The Python runtime does not depend on MATLAB. Fixtures are committed only when
they are small and stable enough for regular `uv run pytest` runs. Large
generated parity snapshots are ignored and should stay local; tests that need
a missing fixture fail directly with the missing data file as the reason.

Current fixtures cover Legendre helpers, including extended helper parity,
basic circle chunking, core geometry/domain/chunkgraph helpers, chunker
transforms, point-kernel evaluators, MATLAB `@kernel` object algebra/factories,
Green helpers, biharmonic `bhgreen`-derived selectors, and dense/native
operator paths. For known MATLAB fixture inconsistencies, tests prefer
component-wise references over expected failures when the intended Python
behavior is clear.

`devtools_easy.mat` is a generated local snapshot produced by
`scripts/matlab/generate_devtools_easy_fixture.m`. It starts the
`external/chunkie-matlab/devtools/test` parity track with compact outputs from
the easiest devtools tests. It is intentionally not tracked because it is large
and regenerated from scripts; tests that need it fail until it exists locally.

For manual inspection, `scripts/generate_devtools_easy_python_fixture.py`
generates `devtools_easy_python.npz` from the live Python package using the
same saved MATLAB inputs. The pytest suite still recomputes Python outputs live,
and the generated `.npz` is intentionally not tracked.

`quadggq.mat` remains tracked because it covers the singular quadrature
behavior that is part of the package surface, including native quadrature, GGQ
self/near/sparse special blocks, and adaptive close quadrature. Runtime GGQ
table data lives in `src/chunkie/data/quadggq` as packaged NumPy assets.

`geometry_core.mat` remains tracked because it is a compact fixture for the
implemented I GEOMETRY surface, excluding `chunkerfit` and smoother workflows.
