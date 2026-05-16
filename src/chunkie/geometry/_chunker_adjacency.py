"""Adjacency helpers shared by chunker storage and constructors."""

from __future__ import annotations

import numpy as np


def _remap_adjacency(adjs: np.ndarray, inds: np.ndarray) -> np.ndarray:
    inverse = {old + 1: new + 1 for new, old in enumerate(inds)}
    out = adjs.copy()
    for old_label, new_label in inverse.items():
        out[adjs == old_label] = new_label
    return out
