"""Small deprecation helpers for legacy public adapter inputs."""

from __future__ import annotations

import warnings
from typing import Any

_NORMALIZED_MARKERS = {
    "_chunkie_legacy_options_warned",
    "_chunkie_normalized_geometry_options",
    "_chunkie_normalized_operator_options",
}


def warn_legacy_options(
    options: dict[str, Any] | None,
    api_name: str,
    *,
    stacklevel: int = 3,
) -> dict[str, Any]:
    """Copy legacy option dictionaries and warn unless they are internal adapters."""

    if options is None:
        return {}
    copied = dict(options)
    if not any(bool(copied.get(marker, False)) for marker in _NORMALIZED_MARKERS):
        warnings.warn(
            f"{api_name} option dictionaries are deprecated; use keyword-only arguments instead",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        copied["_chunkie_legacy_options_warned"] = True
    return copied
