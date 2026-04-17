"""Example potentials V(x, y) for the 2D Schrödinger demo. All vectorised
over numpy arrays."""

from __future__ import annotations

import numpy as np


def V_harmonic_oscillator(x, y, *, omega: float = 50.0,
                          x0: float = 0.5, y0: float = 0.5):
    """2D harmonic oscillator  V = ½ ω² ((x - x₀)² + (y - y₀)²) centred at
    (x₀, y₀). With `ω = 50` the ground-state Gaussian has width `1/√ω ≈ 0.14`
    which fits comfortably inside `[0, 1]²`, so the Dirichlet-box eigenvalues
    are exponentially close to the analytic `E_{n_x, n_y} = (n_x + n_y + 1) ω`.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return 0.5 * omega ** 2 * ((x - x0) ** 2 + (y - y0) ** 2)


def V_double_well(x, y, *, barrier: float = 30.0, width: float = 0.08,
                  sep: float = 0.25):
    """Two Gaussian wells centred at (0.5 ± sep/2, 0.5), depth `-barrier`
    (i.e. the potential has minima at the well centres and ~0 far away).
    The wells are connected by a tunnelling barrier — the splitting of the
    lowest two eigenvalues measures the tunnelling amplitude.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    g = lambda cx, cy: np.exp(
        -((x - cx) ** 2 + (y - cy) ** 2) / (2 * width ** 2)
    )
    return -barrier * (g(0.5 - sep / 2, 0.5) + g(0.5 + sep / 2, 0.5))


def V_random(x, y, *, goo: np.ndarray, M: int = 64, scale: float = 1000.0):
    """Random rough potential: bilinear interpolation of a `(M+2)×(M+2)`
    random field times `scale`. Useful for Anderson-localisation demos.

    `goo` can be generated with
        >>> goo = np.random.default_rng(0).standard_normal((M+2, M+2))
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ti = np.floor(M * x).astype(np.int64)
    si = np.floor(M * y).astype(np.int64)
    ti = np.clip(ti, 0, M)
    si = np.clip(si, 0, M)
    a_t = M * x - ti
    a_s = M * y - si
    bilinear = ((1 - a_s) * (1 - a_t) * goo[ti, si]
                + (1 - a_s) * a_t * goo[ti + 1, si]
                + a_s * (1 - a_t) * goo[ti, si + 1]
                + a_s * a_t * goo[ti + 1, si + 1])
    return scale * bilinear


def V_periodic_lattice(x, y, *, depth: float = 100.0, n_periods: int = 8):
    """Bloch-style periodic lattice potential
       V(x, y) = −depth · (cos²(π n_periods x) + cos²(π n_periods y)).
    The bound states for deep wells localise on lattice sites.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return -depth * (np.cos(np.pi * n_periods * x) ** 2
                     + np.cos(np.pi * n_periods * y) ** 2)
