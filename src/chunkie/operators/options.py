"""Operator option normalization and typed option accessors."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

_NORMALIZED_OPTIONS_MARKER = "_chunkie_normalized_operator_options"


def _normalize_public_options(
    raw_options: dict[str, Any] | None,
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    l2scale: bool | None = None,
    dval: ArrayLike | float | complex | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    flam_occupancy: int | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    force_adaptive: bool | None = None,
    corrections: bool | None = None,
    correction_matrix: ArrayLike | sparse.spmatrix | None = None,
    side: str | None = None,
    near_factor: float | None = None,
    rcip: Any | None = None,
    return_rcip: bool | None = None,
    rcip_context: Any | None = None,
    rcip_subdivisions: int | float | None = None,
    rcip_save_depth: int | None = None,
    rcip_eval_depth: int | None = None,
    rcip_vertices: ArrayLike | None = None,
    rcip_ignore_vertices: ArrayLike | None = None,
) -> dict[str, Any]:
    if raw_options is None:
        options: dict[str, Any] = {}
    elif bool(raw_options.get(_NORMALIZED_OPTIONS_MARKER, False)):
        options = dict(raw_options)
    else:
        warnings.warn(
            "operator option dictionaries are deprecated; use keyword-only arguments instead",
            DeprecationWarning,
            stacklevel=3,
        )
        options = dict(raw_options)
    _set_option(options, "acceleration", acceleration)
    _set_option(options, "usepquad", use_panel_quadrature)
    _set_option(options, "l2scale", l2scale)
    _set_option(options, "dval", dval)
    _set_option(options, "tol", tol)
    _set_option(options, "flamtype", flam_type)
    _set_option(options, "occ", flam_occupancy)
    _set_option(options, "rank_or_tol", rank_or_tol)
    _set_option(options, "useproxy", proxy)
    _set_option(options, "forceadap", force_adaptive)
    _set_option(options, "corrections", corrections)
    _set_option(options, "cormat", correction_matrix)
    _set_option(options, "side", side)
    _set_option(options, "fac", near_factor)
    _set_option(options, "rcip", rcip)
    _set_option(options, "return_rcip", return_rcip)
    _set_option(options, "rcip_context", rcip_context)
    _set_option(options, "nsub", rcip_subdivisions)
    _set_option(options, "rcip_savedepth", rcip_save_depth)
    _set_option(options, "rcip_eval_depth", rcip_eval_depth)
    _set_option(options, "rcip_vertices", rcip_vertices)
    _set_option(options, "rcip_ignore_vertices", rcip_ignore_vertices)
    if quadrature is not None:
        qmode = str(quadrature).lower()
        if qmode == "smooth":
            options["forcesmooth"] = True
        elif qmode == "adaptive":
            options["forceadap"] = True
            options["adaptive_correction"] = True
        elif qmode != "auto":
            options["sing"] = qmode
    options[_NORMALIZED_OPTIONS_MARKER] = True
    return options


def _set_option(options: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        options[key] = value


@dataclass(frozen=True)
class _OperatorOptions:
    raw: dict[str, Any]

    @classmethod
    def from_any(cls, options: dict[str, Any] | _OperatorOptions | None) -> _OperatorOptions:
        if isinstance(options, cls):
            return options
        return cls({} if options is None else dict(options))

    @property
    def acceleration(self) -> str:
        value = self.raw.get("acceleration", "dense")
        if value is None:
            return "dense"
        acceleration = str(value).lower()
        if acceleration not in {"dense", "fmm", "flam"}:
            raise ValueError("acceleration must be one of 'dense', 'fmm', or 'flam'")
        return acceleration

    def flag(self, name: str, default: bool = False) -> bool:
        return _option_bool(self.raw.get(name, default))

    @property
    def l2scale(self) -> bool:
        return self.flag("l2scale")

    @property
    def flamtype(self) -> str:
        return str(self.raw.get("flamtype", "rskelf")).lower()

    @property
    def flam_occ(self) -> int:
        return int(self.raw.get("occ", 200))

    @property
    def flam_rank_or_tol(self) -> int | float:
        value = self.raw.get("rank_or_tol", self.raw.get("eps", self.raw.get("tol", 1.0e-14)))
        value_float = float(value)
        return int(value) if value_float.is_integer() and value_float >= 1 else value_float

    def flam_options(self, *, store_default: str | None = None) -> dict[str, Any]:
        raw_options = {
            "verb": int(self.flag("verb")),
            "lvlmax": self.raw.get("lvlmax", np.inf),
        }
        if store_default is not None:
            raw_options["store"] = self.raw.get("store", store_default)
        return raw_options

    def fmm_tol(self, default: float = 1.0e-12) -> float:
        return float(self.raw.get("eps", self.raw.get("tol", default)))

    def uses_special_quadrature(self, kernel: Any) -> bool:
        if self.flag("forcesmooth") or self.flag("usesmooth"):
            return False
        if self.flag("forceadap"):
            return True
        return getattr(kernel, "sing", "") in {"log", "pv", "hs"}

    def special_quadrature_type(self, kernel: Any) -> str:
        qtype = str(self.raw.get("sing", getattr(kernel, "sing", "log") or "log")).lower()
        return "log" if qtype == "smooth" else qtype


def _option_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)
