from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class AbsOrRelMetrics:
    max_abs_error: float
    max_rel_error: float
    max_scaled_error: float


def assert_abs_or_rel_close(
    actual: ArrayLike,
    expected: ArrayLike,
    *,
    abs_tol: float,
    rel_tol: float,
    label: str = "values",
    test_metrics: Any | None = None,
    equal_nan: bool = False,
) -> AbsOrRelMetrics:
    """Assert that each point passes either an absolute or relative tolerance."""

    if abs_tol < 0 or rel_tol < 0:
        raise ValueError("abs_tol and rel_tol must be nonnegative")

    actual_array, expected_array = np.broadcast_arrays(np.asarray(actual), np.asarray(expected))
    abs_error, rel_error, scaled_error, pass_mask = _pointwise_errors(
        actual_array,
        expected_array,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        equal_nan=equal_nan,
    )
    metrics = AbsOrRelMetrics(
        max_abs_error=_finite_or_inf_max(abs_error),
        max_rel_error=_finite_or_inf_max(rel_error),
        max_scaled_error=_finite_or_inf_max(scaled_error),
    )

    if test_metrics is not None:
        test_metrics.record(f"{label}_max_abs_error", metrics.max_abs_error)
        test_metrics.record(f"{label}_max_rel_error", metrics.max_rel_error)
        test_metrics.record(f"{label}_max_scaled_error", metrics.max_scaled_error)

    if not bool(np.all(pass_mask)):
        failing = np.argwhere(~pass_mask)
        first_index = tuple(int(index) for index in failing[0]) if failing.size else ()
        actual_value = actual_array[first_index] if first_index else actual_array.item()
        expected_value = expected_array[first_index] if first_index else expected_array.item()
        raise AssertionError(
            f"{label} are not close under abs-or-rel tolerance; "
            f"first failing index={first_index}, actual={actual_value!r}, expected={expected_value!r}, "
            f"abs_err={abs_error[first_index]!r}, rel_err={rel_error[first_index]!r}, "
            f"abs_tol={abs_tol!r}, rel_tol={rel_tol!r}, "
            f"max_abs_error={metrics.max_abs_error!r}, "
            f"max_rel_error={metrics.max_rel_error!r}, "
            f"max_scaled_error={metrics.max_scaled_error!r}"
        )

    return metrics


def abs_or_rel_metrics(
    actual: ArrayLike,
    expected: ArrayLike,
    *,
    abs_tol: float,
    rel_tol: float,
    equal_nan: bool = False,
) -> AbsOrRelMetrics:
    actual_array, expected_array = np.broadcast_arrays(np.asarray(actual), np.asarray(expected))
    abs_error, rel_error, scaled_error, _ = _pointwise_errors(
        actual_array,
        expected_array,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        equal_nan=equal_nan,
    )
    return AbsOrRelMetrics(
        max_abs_error=_finite_or_inf_max(abs_error),
        max_rel_error=_finite_or_inf_max(rel_error),
        max_scaled_error=_finite_or_inf_max(scaled_error),
    )


def _pointwise_errors(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    abs_tol: float,
    rel_tol: float,
    equal_nan: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        exact = actual == expected
        if equal_nan:
            exact = exact | (np.isnan(actual) & np.isnan(expected))

        abs_error = np.where(exact, 0.0, np.abs(actual - expected))
        expected_abs = np.abs(expected)

        rel_error = np.full(abs_error.shape, math.inf, dtype=float)
        np.divide(abs_error, expected_abs, out=rel_error, where=expected_abs != 0)
        rel_error = np.where(abs_error == 0, 0.0, rel_error)

        abs_scaled = np.full(abs_error.shape, math.inf, dtype=float)
        abs_scaled = abs_error / abs_tol if abs_tol > 0 else np.where(abs_error == 0, 0.0, abs_scaled)

        rel_scaled = np.full(abs_error.shape, math.inf, dtype=float)
        rel_scaled = rel_error / rel_tol if rel_tol > 0 else np.where(rel_error == 0, 0.0, rel_scaled)

        scaled_error = np.where(exact, 0.0, np.minimum(abs_scaled, rel_scaled))
        pass_mask = exact | (abs_error <= abs_tol) | (rel_error <= rel_tol)

    return abs_error, rel_error, scaled_error, pass_mask


def _finite_or_inf_max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if np.any(np.isposinf(values)):
        return math.inf
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.max(finite))
