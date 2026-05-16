"""Private option and curve-output helpers for chunker construction."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from ._chunker_pref import ChunkerPref

_LEGACY_OPTIONS_MARKER = "_chunkie_normalized_geometry_options"


def _legacy_options(options: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if options is None:
        return {_LEGACY_OPTIONS_MARKER: True}
    if bool(options.get(_LEGACY_OPTIONS_MARKER, False)):
        return dict(options)
    label = name.removesuffix(" options")
    warnings.warn(
        f"{label} option dictionaries are deprecated; use keyword-only arguments instead",
        DeprecationWarning,
        stacklevel=3,
    )
    normalized = dict(options)
    normalized[_LEGACY_OPTIONS_MARKER] = True
    return normalized


def _set_option(options: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        options[key] = value


def _pref_with_order(pref: ChunkerPref | dict[str, Any] | None, order: int | None) -> ChunkerPref:
    p = ChunkerPref.from_any(pref)
    if order is None:
        return p
    return ChunkerPref(p.nchmax, int(order), p.dim, p.nchstor, p.verttol)


def _curve_outputs(fcurve: Callable[[np.ndarray], Any], t: np.ndarray) -> tuple[np.ndarray, ...]:
    raw = fcurve(t)
    if isinstance(raw, tuple):
        outs = raw
    else:
        outs = (raw,)
    return tuple(np.asarray(out, dtype=float).reshape(np.asarray(out).shape[0], -1) for out in outs)
