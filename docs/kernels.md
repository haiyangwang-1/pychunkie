# Kernels

`chunkie.kernels` owns direct PDE formulas and singularity metadata. Kernels
return unweighted tensor values:

```text
kernel_values[o, i, t, s]
```

where `o` is output component, `i` is input component, `t` is target point, and
`s` is source point.

The detailed mathematical reference for singularity extraction lives in
`docs/singularity-extraction.md`.

## Singularity Metadata

The rewrite uses operational singularity metadata rather than string labels.
Metadata is expressed as a Laplace-basis expansion built from:

- `G = -log(rho) / (2*pi)`
- `G_a = d_xa G`
- `G_ab = d_xa d_xb G`

The coding track starts with a conservative data model that can represent
tensor coefficients. Exact formulas are added only with smooth-remainder tests.

Required coverage:

- Laplace selectors: exact metadata for `s`, `sg`, `sp`, `d`, `dg`, and `dp`.
- Helmholtz selectors: local log split based on the Laplace basis.
- Biharmonic, Stokes, and elasticity: componentwise metadata before special
  quadrature consumes those kernels.

## Math Track Notes

The singularity design track recommends these formulas as the implementation
target:

- Laplace `s`: `G`.
- Laplace `sg[a]`: `G_a`.
- Laplace `sp`: `n_t[a] G_a`.
- Laplace `d`: `-n_s[a] G_a`.
- Laplace `dg[a]`: `-n_s[b] G_ab`.
- Laplace `dp`: `-n_t[a] n_s[b] G_ab`.
- Helmholtz uses the exact local log split `J0(k rho) * G`; differentiated
  selectors should be produced by product-rule helpers rather than ad hoc
  selector formulas.

Current implementation status:

- Laplace metadata is active for `s`, `sg`, `sp`, `d`, `dg`, and `dp`.
- Helmholtz metadata is active for the same selectors, using product-rule terms
  from `J0(k rho) * G`.
- Kernel algebra scales singular expansions and canonicalizes exact
  scalar/matrix coefficient cancellation.
- Biharmonic metadata is active for value, gradient, first normal derivatives,
  and Hessian selectors using $B=-(\rho^2/4)G$ product-rule terms.
- Stokes velocity single- and double-layer metadata is active for `s` and `d`,
  using the Stokeslet split through `G` and `B_ij`.
- Elasticity single-displacement metadata is active for `s`, using
  `2*pi*(gamma-beta)*delta_ij*G + 4*pi*gamma*B_ij`.
