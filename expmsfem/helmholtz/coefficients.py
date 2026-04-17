"""Helmholtz coefficient functions. Mirrors Matlab helmholtz/case1/Exp/{afun,
betafun,ffun,gfun}.m and the inferred vfun=1 (vfun is referenced in
elementstiff1.m but not present in the repo; case1 uses constant wave speed)."""

from __future__ import annotations

import numpy as np


def afun(x, y):
    """Diffusion coefficient a(x, y) = 1 (helmholtz/case1)."""
    return np.ones_like(np.asarray(x) * np.asarray(y))


def vfun(x, y):
    """Wave speed v(x, y) = 1 (inferred; Matlab references vfun but does not
    ship a vfun.m for case1)."""
    return np.ones_like(np.asarray(x) * np.asarray(y))


def betafun(x, y):
    """Impedance-boundary indicator: 1 on ∂Ω, 0 inside.

    Matches Matlab betafun.m: `f = 1 iff (1-x)(1-y)·x·y == 0`.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    prod = (1 - x) * (1 - y) * x * y
    return np.where(prod == 0.0, 1.0, 0.0)


def ffun(x, y):
    """Interior source: f = 0 in helmholtz/case1."""
    return np.zeros_like(np.asarray(x) * np.asarray(y))


# Matlab gfun.m uses per-side coefficients a = [0.2, 1.6, 1.8, 0.4, 0]
# indexed as:   y=0 → 1 (bottom),
#               x=1 → 2 (right),
#               y=1 → 3 (top),
#               x=0 → 4 (left),
#               interior → 5 (coefficient zero).
_G_SIDE_COEFS = np.array([0.2, 1.6, 1.8, 0.4, 0.0])


def _g_side_index(x: float, y: float) -> int:
    """Return the 1-indexed side identifier used by Matlab gfun.m."""
    # Conditions evaluated in Matlab order.
    if x == 0:
        return 4
    if x == 1:
        return 2
    if y == 0:
        return 1
    if y == 1:
        return 3
    return 5


def gfun(x, y, k0: float):
    """Impedance boundary data (outgoing plane wave). Vectorised over x, y.

    g(x, y) = -i k₀ · a_side · exp(-0.6 i k₀ x - 0.8 i k₀ y)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # Side identifier map: 0 (bottom) if y==0, 1 (right) if x==1,
    # 2 (top) if y==1, 3 (left) if x==0, 4 (interior) otherwise.
    # Matlab order of evaluation chooses `x==0` first, then x==1, then y==0, then y==1.
    # We replicate that precedence here.
    idx = np.full(x.shape, 4, dtype=np.int64)
    idx = np.where(y == 1, 2, idx)
    idx = np.where(y == 0, 0, idx)
    idx = np.where(x == 1, 1, idx)
    idx = np.where(x == 0, 3, idx)
    coefs = np.array([0.2, 1.6, 1.8, 0.4, 0.0])
    a_side = coefs[idx]
    return (-1j * k0) * a_side * np.exp(-0.6j * k0 * x - 0.8j * k0 * y)
