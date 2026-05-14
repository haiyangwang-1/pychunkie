from __future__ import annotations

import inspect
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("pychunkie metrics")
    group.addoption(
        "--test-log",
        action="store",
        default="test_logs/pytest_metrics.md",
        help="Markdown path for the generated per-test metrics log.",
    )
    group.addoption(
        "--test-log-json",
        action="store",
        default="test_logs/pytest_metrics.jsonl",
        help="JSONL path for the generated per-test metrics log.",
    )
    group.addoption(
        "--no-test-log",
        action="store_true",
        help="Disable generated pytest metrics logs.",
    )
    group.addoption(
        "--no-allclose-metrics",
        action="store_true",
        help="Disable automatic max absolute/relative error capture from numpy.testing.assert_allclose.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config._pychunkie_test_reports = []  # type: ignore[attr-defined]


@pytest.fixture
def test_metrics(request: pytest.FixtureRequest) -> TestMetrics:
    return TestMetrics(request.node)


@pytest.fixture(autouse=True)
def _capture_allclose_metrics(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    if request.config.getoption("--no-allclose-metrics"):
        yield
        return

    original = np.testing.assert_allclose

    def wrapped_assert_allclose(actual, desired, *args, **kwargs):
        _record_allclose_metrics(request.node, actual, desired)
        return original(actual, desired, *args, **kwargs)

    monkeypatch.setattr(np.testing, "assert_allclose", wrapped_assert_allclose)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return

    metrics = _metrics_for(item)
    property_metrics = {key: value for key, value in report.user_properties}
    combined_metrics = {**property_metrics, **metrics}

    existing_property_names = {key for key, _ in report.user_properties}
    for key, value in metrics.items():
        if key not in existing_property_names:
            report.user_properties.append((key, value))

    item.config._pychunkie_test_reports.append(  # type: ignore[attr-defined]
        {
            "nodeid": item.nodeid,
            "description": _description_for(item),
            "outcome": report.outcome,
            "duration_s": report.duration,
            "metrics": _json_safe(combined_metrics),
        }
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if config.getoption("--no-test-log") or getattr(config.option, "collectonly", False):
        return

    reports = config._pychunkie_test_reports  # type: ignore[attr-defined]
    root = Path(str(config.rootpath))
    md_path = _resolve_log_path(root, config.getoption("--test-log"))
    json_path = _resolve_log_path(root, config.getoption("--test-log-json"))

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as stream:
            for report in reports:
                stream.write(json.dumps(report, sort_keys=True, allow_nan=False))
                stream.write("\n")

    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_format_markdown_log(reports, exitstatus), encoding="utf-8")


class TestMetrics:
    def __init__(self, node: pytest.Item) -> None:
        self._node = node

    def record(self, key: str, value: Any) -> None:
        _metrics_for(self._node)[key] = _json_safe(value)

    def error(self, label: str, actual: Any, expected: Any) -> tuple[float, float] | None:
        errors = _error_metrics(actual, expected)
        if errors is None:
            return None
        max_abs_error, max_rel_error = errors
        self.record(f"{label}_max_abs_error", max_abs_error)
        self.record(f"{label}_max_rel_error", max_rel_error)
        return max_abs_error, max_rel_error


def _metrics_for(item: pytest.Item) -> dict[str, Any]:
    metrics = getattr(item, "_pychunkie_test_metrics", None)
    if metrics is None:
        metrics = {}
        item._pychunkie_test_metrics = metrics
    return metrics


def _record_allclose_metrics(item: pytest.Item, actual: Any, expected: Any) -> None:
    errors = _error_metrics(actual, expected)
    if errors is None:
        return

    max_abs_error, max_rel_error = errors
    metrics = _metrics_for(item)
    metrics["allclose_checks"] = int(metrics.get("allclose_checks", 0)) + 1
    metrics["allclose_max_abs_error"] = max(
        float(metrics.get("allclose_max_abs_error", 0.0)), max_abs_error
    )
    metrics["allclose_max_rel_error"] = max(
        float(metrics.get("allclose_max_rel_error", 0.0)), max_rel_error
    )


def _error_metrics(actual: Any, expected: Any) -> tuple[float, float] | None:
    try:
        actual_arr, expected_arr = np.broadcast_arrays(np.asarray(actual), np.asarray(expected))
        dtype = np.result_type(actual_arr, expected_arr)
    except Exception:
        return None

    if not np.issubdtype(dtype, np.number):
        return None

    try:
        with np.errstate(all="ignore"):
            diff = np.abs(actual_arr - expected_arr)
            expected_abs = np.abs(expected_arr)
    except Exception:
        return None

    finite_diff = diff[np.isfinite(diff)]
    if finite_diff.size == 0:
        return None

    max_abs_error = float(np.max(finite_diff))
    finite_expected = expected_abs[np.isfinite(expected_abs)]
    reference_scale = float(np.max(finite_expected)) if finite_expected.size else 0.0
    if reference_scale == 0.0:
        max_rel_error = 0.0 if max_abs_error == 0.0 else math.inf
    else:
        max_rel_error = max_abs_error / reference_scale
    return max_abs_error, max_rel_error


def _description_for(item: pytest.Item) -> str:
    obj = getattr(item, "obj", None)
    doc = inspect.getdoc(obj)
    if doc:
        return doc.splitlines()[0]

    name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
    if name.startswith("test_"):
        name = name.removeprefix("test_")
    return name.replace("_", " ")


def _resolve_log_path(root: Path, option_value: str | None) -> Path | None:
    if not option_value:
        return None
    path = Path(option_value)
    return path if path.is_absolute() else root / path


def _format_markdown_log(reports: list[dict[str, Any]], exitstatus: int) -> str:
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    counts = Counter(report["outcome"] for report in reports)
    total_duration = sum(float(report["duration_s"]) for report in reports)

    lines = [
        "# Pytest Metrics Log",
        "",
        f"Generated: {generated_at}",
        f"Exit status: {exitstatus}",
        f"Tests recorded: {len(reports)}",
        f"Total call time: {_format_number(total_duration)} s",
        "",
        "| Outcome | Count |",
        "| --- | ---: |",
    ]
    for outcome, count in sorted(counts.items()):
        lines.append(f"| {_escape_markdown(outcome)} | {count} |")

    lines.extend(
        [
            "",
            "| Test | Description | Outcome | Time (s) | Allclose | Max abs err | Max rel err | Other metrics |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for report in reports:
        metrics = report["metrics"]
        other = {
            key: value
            for key, value in metrics.items()
            if key not in {"allclose_checks", "allclose_max_abs_error", "allclose_max_rel_error"}
        }
        lines.append(
            "| {test} | {description} | {outcome} | {duration} | {checks} | {abs_err} | {rel_err} | {other} |".format(
                test=_escape_markdown(report["nodeid"]),
                description=_escape_markdown(report["description"]),
                outcome=_escape_markdown(report["outcome"]),
                duration=_format_number(report["duration_s"]),
                checks=_format_int(metrics.get("allclose_checks")),
                abs_err=_format_number(metrics.get("allclose_max_abs_error")),
                rel_err=_format_number(metrics.get("allclose_max_rel_error")),
                other=_escape_markdown(_format_other_metrics(other)),
            )
        )

    lines.append("")
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        if value.size <= 8:
            return _json_safe(value.tolist())
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:.3e}"


def _format_int(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _format_other_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    return ", ".join(
        f"{key}={_format_metric_value(value)}" for key, value in sorted(metrics.items())
    )


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
