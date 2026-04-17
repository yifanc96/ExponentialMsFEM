"""Heterogeneous coefficient fields a(x, y).

The `afun_periodic` matches the Matlab elliptic/periodic/Exp/afun.m exactly.
"""

import numpy as np


def afun_periodic(x, y):
    """Multi-scale periodic coefficient from Matlab afun.m.

    a(t,s) = 1/6 * (
        (1.1 + sin(2π t/ε1)) / (1.1 + sin(2π s/ε1))
      + (1.1 + sin(2π s/ε2)) / (1.1 + cos(2π t/ε2))
      + (1.1 + cos(2π t/ε3)) / (1.1 + sin(2π s/ε3))
      + (1.1 + sin(2π s/ε4)) / (1.1 + cos(2π t/ε4))
      + (1.1 + cos(2π t/ε5)) / (1.1 + sin(2π s/ε5))
      + sin(4 s^2 t^2) + 1
    )

    ε1..ε5 = 1/5, 1/13, 1/17, 1/31, 1/65.
    """
    t = np.asarray(x)
    s = np.asarray(y)
    eps = (1 / 5, 1 / 13, 1 / 17, 1 / 31, 1 / 65)
    pi2 = 2 * np.pi
    term = (
        (1.1 + np.sin(pi2 * t / eps[0])) / (1.1 + np.sin(pi2 * s / eps[0]))
        + (1.1 + np.sin(pi2 * s / eps[1])) / (1.1 + np.cos(pi2 * t / eps[1]))
        + (1.1 + np.cos(pi2 * t / eps[2])) / (1.1 + np.sin(pi2 * s / eps[2]))
        + (1.1 + np.sin(pi2 * s / eps[3])) / (1.1 + np.cos(pi2 * t / eps[3]))
        + (1.1 + np.cos(pi2 * t / eps[4])) / (1.1 + np.sin(pi2 * s / eps[4]))
        + np.sin(4 * s**2 * t**2)
        + 1.0
    )
    return term / 6.0


def afun_constant(x, y):
    """a ≡ 1 baseline."""
    return np.ones_like(np.asarray(x) * np.asarray(y))


# -----------------------------------------------------------------------------
# Random heterogeneous coefficient — Matlab elliptic/Random/Exp/afun.m
# -----------------------------------------------------------------------------


def generate_random_field(M: int = 128, seed: int = 0) -> np.ndarray:
    """Matches Matlab `rando.m`: goo = |randn(MxM)| + 0.5.

    Returned array has shape (M+2, M+2) — padded by 1 on the max edges so that
    bilinear interpolation at t=1 / s=1 stays in-bounds (Matlab reads
    `goo(t1, s1)` through `goo(t2, s2)` with t2 = floor(M*t) + 2, which at t=1
    is index M+2 (1-indexed) = M+1 (0-indexed)).
    """
    rng = np.random.default_rng(seed)
    return np.abs(rng.standard_normal((M + 2, M + 2))) + 0.5


def afun_random(goo: np.ndarray, M: int = 128):
    """Factory: returns a callable a(t, s) that bilinearly interpolates `goo`
    at continuous (t, s) ∈ [0, 1]². Mirrors Matlab elliptic/Random/Exp/afun.m.

    `goo` must have shape at least (M+2, M+2).
    """
    assert goo.shape[0] >= M + 2 and goo.shape[1] >= M + 2

    def a(t, s):
        t = np.asarray(t, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        # Matlab: t1 = floor(M*t) + 1 (1-indexed) ⇒ 0-indexed t_i = floor(M*t)
        ti = np.floor(M * t).astype(np.int64)
        si = np.floor(M * s).astype(np.int64)
        # Clamp for the edge t = 1 exactly (then ti = M, alpha_t = 0 so no issue)
        ti = np.clip(ti, 0, M)
        si = np.clip(si, 0, M)
        alpha_t = M * t - ti
        alpha_s = M * s - si
        # Matlab formula re-written with 0-indexed goo:
        #   a = (1-alpha_s)(1-alpha_t) goo[ti,si]
        #     + (1-alpha_s) alpha_t   goo[ti+1,si]
        #     + alpha_s (1-alpha_t)   goo[ti,si+1]
        #     + alpha_s alpha_t       goo[ti+1,si+1]
        return (
            (1 - alpha_s) * (1 - alpha_t) * goo[ti, si]
            + (1 - alpha_s) * alpha_t * goo[ti + 1, si]
            + alpha_s * (1 - alpha_t) * goo[ti, si + 1]
            + alpha_s * alpha_t * goo[ti + 1, si + 1]
        )

    return a


# -----------------------------------------------------------------------------
# High-contrast coefficient — Matlab elliptic/Highcontrast/Exp/afun.m
# -----------------------------------------------------------------------------


# Pre-computed 49 inclusion centres on the 7×7 grid 0.2..0.8 step 0.1
_HC_CENTERS = np.stack(
    np.meshgrid(np.arange(0.2, 0.8001, 0.1),
                np.arange(0.2, 0.8001, 0.1), indexing="xy"),
    axis=-1,
).reshape(-1, 2)  # shape (49, 2)

HIGH_CONTRAST_RADIUS = 0.015
HIGH_CONTRAST_CONTRAST = 2 ** 6  # 64×


def afun_highcontrast(x, y):
    """Piecewise-constant coefficient with 49 circular inclusions.

    Matches Matlab elliptic/Highcontrast/Exp/afun.m: a = 1 outside every
    inclusion, a = 64 inside any of 49 disks of radius 0.015 centred on the
    7×7 grid at positions (0.2, 0.3, ..., 0.8) × (0.2, 0.3, ..., 0.8).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    shape = x.shape
    x_flat = x.ravel()
    y_flat = y.ravel()
    # Distance to every centre: (n_points, 49)
    dx = x_flat[:, None] - _HC_CENTERS[None, :, 0]
    dy = y_flat[:, None] - _HC_CENTERS[None, :, 1]
    d2 = dx * dx + dy * dy
    min_d = np.sqrt(d2.min(axis=1))
    a = np.where(
        min_d > HIGH_CONTRAST_RADIUS,
        1.0,
        float(HIGH_CONTRAST_CONTRAST),
    )
    return a.reshape(shape)


# -----------------------------------------------------------------------------
# Perforated / complicated-geometry coefficient
# -----------------------------------------------------------------------------


def afun_perforated(hole_centers: np.ndarray, hole_radius: float,
                    a_out: float = 1.0, a_in: float = 1e-6):
    """Factory: returns a callable `a(x, y)` that approximates a perforated
    domain via a piecewise coefficient — `a_out` in the bulk material,
    `a_in` (very small by default) inside each hole. Holes are discs of
    radius `hole_radius` centred on `hole_centers` (shape `(n_holes, 2)`).

    With `a_in / a_out → 0` the tiny-diffusion regions become an approximate
    Dirichlet-zero trace on the hole boundaries, giving the classical
    perforated-domain problem in the limit (a standard "fictitious domain"
    reformulation). The ExpMsFEM pipeline handles this without modifying the
    rectangular background mesh — the edge eigenbasis adapts automatically
    to the high-contrast jumps at every hole boundary.
    """
    hole_centers = np.asarray(hole_centers, dtype=np.float64)
    assert hole_centers.ndim == 2 and hole_centers.shape[1] == 2

    def a(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = x.shape
        x_flat = x.ravel()
        y_flat = y.ravel()
        dx = x_flat[:, None] - hole_centers[None, :, 0]
        dy = y_flat[:, None] - hole_centers[None, :, 1]
        d2 = dx * dx + dy * dy
        inside_any = d2.min(axis=1) < hole_radius ** 2
        return np.where(inside_any, a_in, a_out).reshape(shape)

    return a


def default_hole_lattice(n_per_side: int = 4, margin: float = 0.125) -> np.ndarray:
    """Convenience: `(n_per_side**2, 2)` hole centres on a regular lattice
    inset by `margin` from the edges of `[0, 1]²`."""
    pts = np.linspace(margin, 1 - margin, n_per_side)
    X, Y = np.meshgrid(pts, pts, indexing="xy")
    return np.stack([X.ravel(), Y.ravel()], axis=-1)
