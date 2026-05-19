"""Solution field evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chunkie.quadrature import apply_panel_potential

from .config import SystemConfig


@dataclass(frozen=True)
class FieldResult:
    points: object
    values: NDArray[np.generic]
    field: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def evaluate_solution(solution, targets, *, field: str, config=None) -> FieldResult:
    if field not in solution.system.fields:
        raise KeyError(field)
    active_config = SystemConfig() if config is None else config
    if active_config.evaluation_method != "dense":
        raise ValueError("evaluation_method must be 'dense'")
    values = None
    rcip_state = (
        solution.operator.diagnostics.get("rcip") if solution.operator is not None else None
    )
    for layer in solution.system.fields[field]:
        density = solution.densities[layer.density]
        contribution = None
        if rcip_state is not None:
            from .nonsmooth import evaluate_rcip_layer

            contribution = evaluate_rcip_layer(
                layer,
                density,
                targets,
                config=active_config,
                state=rcip_state,
            )
        if contribution is None:
            contribution = layer.coefficient * apply_panel_potential(
                layer.source.pointinfo,
                targets,
                layer.kernel,
                density.component_values,
                close_correction=active_config.close_correction,
                near_rho=active_config.near_rho,
                tolerance=active_config.tolerance,
            )
        values = contribution if values is None else values + contribution
    assert values is not None
    return FieldResult(
        points=targets,
        values=values,
        field=field,
        diagnostics={
            "evaluation": active_config.evaluation_method,
            "close_correction": active_config.close_correction,
            "near_rho": active_config.near_rho,
            "config": active_config,
        },
    )
