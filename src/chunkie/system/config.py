"""System-level configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SystemConfig:
    assembly_method: Literal["dense", "matrix_free"] = "dense"
    solve_method: Literal["dense", "gmres"] = "dense"
    evaluation_method: Literal["dense"] = "dense"
    tolerance: float = 1.0e-12
    max_iterations: int | None = None
    close_correction: bool = True
    near_rho: float = 1.8
    singular_quadrature: Literal["auto", "ggq", "helsing_ojala", "adaptive"] = "auto"
    prefer_helsing_ojala: bool = True
    use_rcip: bool = True
    rcip_subdivisions: int = 20
    rcip_eval_depth: int | None = None

    def __post_init__(self) -> None:
        if self.solve_method not in {"dense", "gmres"}:
            raise ValueError("solve_method must be 'dense' or 'gmres'")
        if self.evaluation_method != "dense":
            raise ValueError("evaluation_method must be 'dense'")
