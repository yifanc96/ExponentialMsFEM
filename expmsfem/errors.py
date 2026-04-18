"""Relative L² and H¹ error norms vs a fine-scale reference."""

from __future__ import annotations

import numpy as np


def rel_l2(u_ref: np.ndarray, u_ms: np.ndarray, M) -> float:
    e = u_ref - u_ms
    num = float(e @ (M @ e))
    den = float(u_ref @ (M @ u_ref))
    return np.sqrt(num / den) if den > 0 else np.inf


def rel_h1(u_ref: np.ndarray, u_ms: np.ndarray, K) -> float:
    e = u_ref - u_ms
    num = float(e @ (K @ e))
    den = float(u_ref @ (K @ u_ref))
    return np.sqrt(num / den) if den > 0 else np.inf
