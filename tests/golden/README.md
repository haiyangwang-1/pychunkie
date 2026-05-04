# Golden Fixtures

This directory stores compact MATLAB-generated fixtures used for parity tests.

Generate them after MATLAB chunkIE is working locally:

```powershell
matlab -batch "run('scripts/matlab/generate_basic_fixtures.m')"
```

The Python runtime does not depend on MATLAB. Fixtures are committed only when
they are small and stable enough for regular `uv run pytest` runs.

Current fixtures cover Legendre helpers, basic circle chunking, chunker
transforms, point-kernel evaluators, and dense/native operator paths. Strict
`xfail` entries in `tests/test_matlab_parity.py` document known MATLAB parity
gaps while keeping the rest of the suite actionable.
