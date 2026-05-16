# Quadrature

`chunkie.quadrature` owns local panel rules only. Global correction insertion,
system assembly, density layout, and backend acceleration belong to
`chunkie.system`.

Implementation order:

1. Dense reference panel quadrature.
2. Adaptive fallback.
3. Helsing-Ojala conversion from Laplace-basis metadata.
4. GGQ parity driven by `SingularityInfo` dispatch.

Helsing-Ojala and GGQ parity are required rewrite milestones.
