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

The current implementation generates rules dynamically instead of copying
MATLAB's tabulated GGQ files. Self-panel rules split the Legendre panel at
each target node and place Gauss-Legendre nodes on both sides, so logarithmic
kernels are never evaluated at the singular source/target coincidence.
Neighbor blocks use an oversampled Gauss-Legendre rule and interpolate source
geometry and density values from the original panel nodes.

`chunkermat`, `chunkermatapply`, `chunkerkerneval`, and `chunkerkernevalmat`
use this special quadrature by default for kernel objects marked with
`sing == "log"` when the target is the same chunker. Pass
`{"forcesmooth": True}` or `{"usesmooth": True}` to force native smooth
quadrature.

This is not yet a byte-for-byte port of MATLAB's precomputed GGQ tables for
principal-value or hypersingular kernels. Those can be added later either by
copying and packaging the MATLAB table data, or by extending the generated
moment-fitting routines for `pv` and `hs` singularities.
