# Golden Fixtures

This directory stores compact MATLAB-generated fixtures used for parity tests.

Generate them after MATLAB chunkIE is working locally:

```powershell
matlab -batch "run('scripts/matlab/generate_basic_fixtures.m')"
```

The Python runtime does not depend on MATLAB. Fixtures are committed only when
they are small and stable enough for regular `uv run pytest` runs.
