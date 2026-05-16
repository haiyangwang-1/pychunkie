# Backends

Acceleration lives under `chunkie.system.backends`.

Required backend milestones:

- FMM-backed matvec and evaluation compared against dense references.
- FLAM apply and solve compared against dense references.

Backends consume kernel metadata and system layout. Kernel objects do not call
FMM or FLAM directly.
