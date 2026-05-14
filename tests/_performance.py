from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar

from _numerical import abs_or_rel_metrics

T = TypeVar("T")


def timed_call(call: Callable[[], T]) -> tuple[T, float]:
    start = perf_counter()
    result = call()
    return result, perf_counter() - start


def record_backend_metrics(
    test_metrics: Any,
    label: str,
    *,
    backend: str,
    problem_size: dict[str, int | float | str],
    elapsed_s: float,
    actual: Any,
    expected: Any,
    abs_tol: float,
    rel_tol: float,
    reference_elapsed_s: float | None = None,
) -> None:
    metrics = abs_or_rel_metrics(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol)
    test_metrics.record(f"{label}_backend", backend)
    test_metrics.record(f"{label}_problem_size", problem_size)
    test_metrics.record(f"{label}_elapsed_s", elapsed_s)
    test_metrics.record(f"{label}_abs_tol", abs_tol)
    test_metrics.record(f"{label}_rel_tol", rel_tol)
    test_metrics.record(f"{label}_max_abs_error", metrics.max_abs_error)
    test_metrics.record(f"{label}_max_rel_error", metrics.max_rel_error)
    test_metrics.record(f"{label}_max_scaled_error", metrics.max_scaled_error)
    if reference_elapsed_s is not None:
        test_metrics.record(f"{label}_reference_elapsed_s", reference_elapsed_s)
        speed_ratio = reference_elapsed_s / elapsed_s if elapsed_s > 0 else float("inf")
        test_metrics.record(f"{label}_speed_ratio", speed_ratio)
