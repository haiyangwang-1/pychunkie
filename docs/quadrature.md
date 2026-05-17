# Quadrature

`chunkie.quadrature` owns local panel rules only. Global correction insertion,
system assembly, density layout, and backend acceleration belong to
`chunkie.system`.

Implementation order:

1. Dense reference panel quadrature.
2. Adaptive fallback.
3. Helsing-Ojala conversion from Laplace-basis metadata.
4. GGQ parity driven by `SingularityInfo` dispatch.

Current helpers:

- `legendre_rule`, `exps`, `pols`, `matrin`, `intmat`, and related helpers
  provide the active Legendre transform/interpolation layer used by dense
  panel rules, adaptive fallback, and old-test migration. These utilities keep
  coefficient-space operations separate from solver-vector layouts.
- `dense_panel_matrix` evaluates uncorrected weighted kernel tensors with shape
  `(output, input, target, source)`.
- `dense_panel_operator_matrix` materializes those tensors into the solver
  layout

$$
\mathrm{row}=o\,N_t+t,\qquad \mathrm{col}=i\,N_s+s.
$$

This operator matrix is an adapter boundary between mathematical kernel tensors
and linear algebra. Smooth singular amplitudes from `SingularityInfo` will be
absorbed into panel data before Helsing-Ojala or GGQ weights are applied.
- `adaptive_panel_matrix` builds a local source-panel matrix by interpolating
  the original panel density and geometry onto recursively accepted Legendre
  subpanels. It is the conservative fallback for close-panel reference values
  before specialized Helsing-Ojala or GGQ rules are wired into the correction
  layer.
- `helsing_ojala_weights` exposes the local log, Cauchy, and derivative
  product weights on one complex source panel.
- `build_helsing_ojala_panel_matrix` currently supports Laplace single-layer
  log weights and Laplace double-layer Cauchy weights, returning local matrices
  on the original panel density nodes.
- `helsing_ojala_log_singular_matrix` consumes log-basis terms from
  `SingularityInfo`, including smooth target/source amplitudes such as the
  Helmholtz $J_0(k\rho)$ factor.
- `setup_ggq` and `ggq_removable_rules` create generated GGQ-style neighbor
  and self split rules with interpolation matrices. MATLAB table-backed parity
  is still upcoming.
- `build_ggq_self_panel_matrix` uses generated self split rules to build a
  Laplace single-layer self-panel matrix on the original source nodes.

Remaining Helsing-Ojala work includes normal-derivative and hypersingular
`SingularityInfo` dispatch. GGQ parity is also a required rewrite milestone.
