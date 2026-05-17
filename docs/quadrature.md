# Quadrature

`chunkie.quadrature` owns local panel rules only. Global correction insertion,
system assembly, density layout, and backend acceleration belong to
`chunkie.system`.

Implementation order:

1. Dense reference panel quadrature.
2. Adaptive fallback.
3. Helsing-Ojala conversion from Laplace-basis metadata.
4. GGQ parity driven by `SingularityInfo` dispatch.

Current dense helpers:

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

Helsing-Ojala and GGQ parity are required rewrite milestones.
