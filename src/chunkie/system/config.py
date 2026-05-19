"""System-level configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SystemConfig:
    assembly_method: Literal["dense", "matrix_free"] = "dense"
    solve_method: Literal["dense", "gmres", "flam"] = "dense"
    evaluation_method: Literal["dense", "fmm"] = "dense"
    tolerance: float = 1.0e-12
    max_iterations: int | None = None
    close_correction: bool = True
    near_factor: float = 1.0
    singular_quadrature: Literal["auto", "ggq", "helsing_ojala", "adaptive"] = "auto"
    prefer_helsing_ojala: bool = True
    use_rcip: bool = True
    rcip_subdivisions: int = 20
    rcip_eval_depth: int | None = None
    fmm_tolerance: float = 1.0e-12
    flam_tolerance: float = 1.0e-12
    flam_occupancy: int = 200
    flam_proxy: bool = True
