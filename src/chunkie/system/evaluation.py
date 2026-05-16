"""Solution field evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chunkie.quadrature import apply_panel_potential


@dataclass(frozen=True)
class FieldResult:
    points: object
    values: NDArray[np.generic]
    field: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def evaluate_solution(solution, targets, *, field: str, config=None) -> FieldResult:
    if field not in solution.system.fields:
        raise KeyError(field)
    values = None
    for layer in solution.system.fields[field]:
        density = solution.densities[layer.density]
        contribution = layer.coefficient * apply_panel_potential(
            layer.source.pointinfo,
            targets,
            layer.kernel,
            density.component_values,
        )
        values = contribution if values is None else values + contribution
    assert values is not None
    return FieldResult(
        points=targets,
        values=values,
        field=field,
        diagnostics={"evaluation": "dense", "config": config},
    )
