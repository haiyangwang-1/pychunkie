# Special Quadrature Status

The Python port now includes `chunkie.chnk.quadggq`, matching the MATLAB
special-quadrature entry points:

- `setup`
- `getlogquad`
- `getremovablequad`
- `diagbuildmat`
- `nearbuildmat`
- `buildmat`
- `logavail`

The implementation vendors MATLAB's tabulated GGQ files as NumPy package data
under `src/chunkie/data/quadggq`. Runtime loading uses `importlib.resources`,
so installed Python packages no longer need `external/chunkie-matlab` in order
to assemble log, principal-value, or hypersingular special quadrature blocks.
`scripts/generate_quadggq_package_data.py` regenerates the `.npz` assets from
an upstream `chunkie/+chnk/+quadggq` checkout.

When a requested order has no vendored table, the Python fallback still
generates split or oversampled Gauss-Legendre rules. Self-panel fallback rules
split the Legendre panel at each target node, so logarithmic kernels are never
evaluated at the singular source/target coincidence. Neighbor fallback blocks
use an oversampled Gauss-Legendre rule and interpolate source geometry and
density values from the original panel nodes.

`chunkermat`, `chunkermatapply`, `chunkerkerneval`, and `chunkerkernevalmat`
use this special quadrature by default for kernel objects marked with
`sing == "log"` when the target is the same chunker. Pass
`{"forcesmooth": True}` or `{"usesmooth": True}` to force native smooth
quadrature.

The packaged data covers the upstream near tables, log self tables used by the
Python loader, and the available PV/HS support tables.
