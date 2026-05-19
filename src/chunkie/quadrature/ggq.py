"""Generalized Gaussian quadrature integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chunkie.geometry import PanelView
from chunkie.geometry.chunker import right_normals
from chunkie.kernels import Kernel

from .legendre import interpolation_matrix, legendre_rule


@dataclass(frozen=True)
class GGQRuleSet:
    neighbor_nodes: NDArray[np.floating]
    neighbor_weights: NDArray[np.floating]
    self_nodes: tuple[NDArray[np.floating], ...]
    self_weights: tuple[NDArray[np.floating], ...]
    neighbor_interpolator: NDArray[np.floating]
    self_interpolators: tuple[NDArray[np.floating], ...]
    singularity: str = "log"


def setup(
    quadrature_order: int,
    singularity: str = "log",
    *,
    nfac_self: int = 2,
    nfac_near: int = 2,
) -> GGQRuleSet:
    """Build generated GGQ-style local rules for one panel order."""

    qtype = singularity.lower()
    if qtype not in {"log", "removable", "pv", "hs"}:
        raise ValueError("GGQ singularity must be one of log, removable, pv, or hs")
    order = int(quadrature_order)
    neighbor_nodes, neighbor_weights = legendre_rule(max(order, int(nfac_near) * order))
    self_nodes, self_weights = getremovablequad(order, nfac=max(1, int(nfac_self)))
    base_nodes, _ = legendre_rule(order)
    return GGQRuleSet(
        neighbor_nodes=neighbor_nodes,
        neighbor_weights=neighbor_weights,
        self_nodes=tuple(self_nodes),
        self_weights=tuple(self_weights),
        neighbor_interpolator=interpolation_matrix(base_nodes, neighbor_nodes),
        self_interpolators=tuple(interpolation_matrix(base_nodes, nodes) for nodes in self_nodes),
        singularity=qtype,
    )


def getremovablequad(
    quadrature_order: int,
    *,
    nfac: int = 1,
) -> tuple[list[NDArray[np.floating]], list[NDArray[np.floating]]]:
    """Return split Gauss rules that avoid each original Legendre node."""

    order = int(quadrature_order)
    base_nodes, _ = legendre_rule(order)
    over_nodes, over_weights = legendre_rule(max(order, int(nfac) * order))
    unit_nodes = 0.5 * (over_nodes + 1.0)
    unit_weights = 0.5 * over_weights
    nodes_by_source: list[NDArray[np.floating]] = []
    weights_by_source: list[NDArray[np.floating]] = []
    for node in base_nodes:
        left_nodes = unit_nodes * (node + 1.0) - 1.0
        left_weights = unit_weights * (node + 1.0)
        right_nodes = unit_nodes * (1.0 - node) + node
        right_weights = unit_weights * (1.0 - node)
        nodes_by_source.append(np.concatenate((left_nodes, right_nodes)))
        weights_by_source.append(np.concatenate((left_weights, right_weights)))
    return nodes_by_source, weights_by_source


def build_ggq_panel_matrix(
    panel: PanelView,
    target,
    kernel: Kernel,
    *,
    rules: GGQRuleSet | None = None,
    self_panel: bool = False,
) -> NDArray[np.generic]:
    """Build a generated GGQ-style panel matrix for self or near targets."""

    active_rules = setup(panel.nodes.size) if rules is None else rules
    if self_panel:
        return build_ggq_self_panel_matrix(panel, kernel, rules=active_rules)
    source = _interpolated_source(
        panel,
        active_rules.neighbor_nodes,
        active_rules.neighbor_weights,
        active_rules.neighbor_interpolator,
    )
    kernel_values = kernel(source, target)
    return np.einsum(
        "oitq,qk,q->oitk",
        kernel_values,
        active_rules.neighbor_interpolator,
        source.flat_weights,
    )


def build_ggq_self_panel_matrix(
    panel: PanelView,
    kernel: Kernel,
    *,
    rules: GGQRuleSet | None = None,
) -> NDArray[np.generic]:
    """Build a generated GGQ self-panel matrix for one source panel."""

    if kernel.family != "laplace" or kernel.selector != "s":
        raise NotImplementedError(
            "generated GGQ self-panel bootstrap currently supports Laplace single layer"
        )
    active_rules = setup(panel.nodes.size) if rules is None else rules
    if active_rules.self_nodes.__len__() != panel.nodes.size:
        raise ValueError("GGQ rule set order must match panel order")

    values_by_target = []
    for target_index, (nodes, weights, interpolator) in enumerate(
        zip(
            active_rules.self_nodes,
            active_rules.self_weights,
            active_rules.self_interpolators,
            strict=True,
        )
    ):
        source = _interpolated_source(panel, nodes, weights, interpolator)
        target = panel.positions[:, target_index : target_index + 1]
        kernel_values = kernel(source, target)
        block = np.einsum("oitq,qk,q->oitk", kernel_values, interpolator, source.flat_weights)
        values_by_target.append(block)
    return np.concatenate(values_by_target, axis=2)


@dataclass(frozen=True)
class _GGQSourceView:
    positions: NDArray[np.floating]
    derivatives: NDArray[np.floating]
    second_derivatives: NDArray[np.floating]
    normals: NDArray[np.floating]
    weights: NDArray[np.floating]

    @property
    def flat_positions(self) -> NDArray[np.floating]:
        return self.positions[:, :, 0]

    @property
    def flat_normals(self) -> NDArray[np.floating]:
        return self.normals[:, :, 0]

    @property
    def flat_weights(self) -> NDArray[np.floating]:
        return self.weights[:, 0]


def _interpolated_source(
    panel: PanelView,
    nodes: NDArray[np.floating],
    reference_weights: NDArray[np.floating],
    interpolator: NDArray[np.floating],
) -> _GGQSourceView:
    positions = np.einsum("ql,rl->rq", interpolator, panel.positions)
    derivatives = np.einsum("ql,rl->rq", interpolator, panel.derivatives)
    second_derivatives = np.einsum("ql,rl->rq", interpolator, panel.second_derivatives)
    speed = np.linalg.norm(derivatives, axis=0)
    normals = (
        right_normals(derivatives[:, :, None])[:, :, 0]
        if derivatives.shape[0] == 2
        else panel.normals
    )
    return _GGQSourceView(
        positions=positions[:, :, None],
        derivatives=derivatives[:, :, None],
        second_derivatives=second_derivatives[:, :, None],
        normals=normals[:, :, None],
        weights=(reference_weights * speed)[:, None],
    )
