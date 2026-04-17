"""Q1 local stiffness and load for the Helmholtz case, per fine cell.

Mirrors Matlab elementstiff1.m and elementstiff2.m. Node ordering
counterclockwise: 0=(xlow,ylow), 1=(xhigh,ylow), 2=(xhigh,yhigh),
3=(xlow,yhigh).

Stiffness entry layout (per fine cell, complex):

    diag (i == j):      (2/3) a  − k² v² h²/9  − i k β_corner[i] h/3
    opposite (|i−j|=2): −(1/3) a − k² v² h²/36
    adjacent (1,2),(2,3),(3,4): −(1/6) a − k² v² h²/18 − i k β1[min(i,j)] h/6
    wrap   (0,3),(3,0): −(1/6) a − k² v² h²/18 − i k β_left h/6

where β1 indexes edges (bot, right, top) and β_left is the fourth edge's
β-indicator. `K_norm` is the Hermitian-positive reference for the energy-like
norm: flip the sign of every k²v² term and drop the impedance terms.
"""

from __future__ import annotations

import numpy as np

from . import coefficients as coefs


# Reference Q1 stiffness (Laplacian part, scale-free on squares).
_C_LAP = np.array(
    [
        [2 / 3, -1 / 6, -1 / 3, -1 / 6],
        [-1 / 6, 2 / 3, -1 / 6, -1 / 3],
        [-1 / 3, -1 / 6, 2 / 3, -1 / 6],
        [-1 / 6, -1 / 3, -1 / 6, 2 / 3],
    ]
)

# Reference Q1 mass (no h² factor).
_C_MASS = np.array(
    [
        [1 / 9, 1 / 18, 1 / 36, 1 / 18],
        [1 / 18, 1 / 9, 1 / 18, 1 / 36],
        [1 / 36, 1 / 18, 1 / 9, 1 / 18],
        [1 / 18, 1 / 36, 1 / 18, 1 / 9],
    ]
)

# Which off-diagonal slots get the impedance contribution and with which
# β-index. Matlab ordering:
#   (i,j) ∈ {(0,1),(1,0)}: adjacent edge (1,2) = bottom   β1[0]
#   (i,j) ∈ {(1,2),(2,1)}: adjacent edge (2,3) = right    β1[1]
#   (i,j) ∈ {(2,3),(3,2)}: adjacent edge (3,4) = top      β1[2]
#   (i,j) ∈ {(0,3),(3,0)}: wrap edge (4,1)   = left       β_left


def element_matrices(xlow, xhigh, ylow, yhigh, k0: float):
    """Compute complex K (4,4) and real K_norm (4,4) for ONE fine Q1 cell.

    Inputs are scalar cell coordinates. Returned arrays are dense.
    """
    x_mid = 0.5 * (xlow + xhigh)
    y_mid = 0.5 * (ylow + yhigh)
    h = xhigh - xlow
    a = float(coefs.afun(x_mid, y_mid))
    v = float(coefs.vfun(x_mid, y_mid))
    # Edge β indicators
    beta_bot = float(coefs.betafun(x_mid, ylow))
    beta_top = float(coefs.betafun(x_mid, yhigh))
    beta_left = float(coefs.betafun(xlow, y_mid))
    beta_right = float(coefs.betafun(xhigh, y_mid))

    beta_corner = np.array(
        [
            beta_bot + beta_left,
            beta_bot + beta_right,
            beta_top + beta_right,
            beta_top + beta_left,
        ]
    )
    beta1 = np.array([beta_bot, beta_right, beta_top])

    lap = a * _C_LAP
    mass = (v ** 2) * (h ** 2) * _C_MASS

    K_norm = lap + (k0 ** 2) * mass     # (elliptic-like real-positive) + mass
    K = lap - (k0 ** 2) * mass          # Helmholtz sign on mass

    # Impedance contributions (only non-zero on boundary cells)
    K = K.astype(np.complex128)
    # Diagonal
    for i in range(4):
        K[i, i] -= 1j * k0 * beta_corner[i] * h / 3
    # Adjacent (0,1),(1,0),(1,2),(2,1),(2,3),(3,2)
    for (i, j, bi) in [
        (0, 1, 0), (1, 0, 0),
        (1, 2, 1), (2, 1, 1),
        (2, 3, 2), (3, 2, 2),
    ]:
        K[i, j] -= 1j * k0 * beta1[bi] * h / 6
    # Wrap (0,3),(3,0) use β_left
    K[0, 3] -= 1j * k0 * beta_left * h / 6
    K[3, 0] -= 1j * k0 * beta_left * h / 6

    return K, K_norm


def element_load(xlow, xhigh, ylow, yhigh, k0: float):
    """Per-cell Q1 load f ∈ ℂ⁴ combining the interior source and the impedance
    boundary data, mirroring Matlab elementstiff2.m."""
    x_mid = 0.5 * (xlow + xhigh)
    y_mid = 0.5 * (ylow + yhigh)
    h = xhigh - xlow
    f_interior = complex(coefs.ffun(x_mid, y_mid)) * (h ** 2) / 4
    g_bot = coefs.gfun(np.array(x_mid), np.array(ylow), k0).item()
    g_top = coefs.gfun(np.array(x_mid), np.array(yhigh), k0).item()
    g_left = coefs.gfun(np.array(xlow), np.array(y_mid), k0).item()
    g_right = coefs.gfun(np.array(xhigh), np.array(y_mid), k0).item()
    f = np.empty(4, dtype=np.complex128)
    f[0] = f_interior + (g_bot + g_left) * h / 2
    f[1] = f_interior + (g_bot + g_right) * h / 2
    f[2] = f_interior + (g_top + g_right) * h / 2
    f[3] = f_interior + (g_top + g_left) * h / 2
    return f


def batched_element_matrices(cell_corners: np.ndarray, k0: float):
    """Vectorised version of `element_matrices` over many fine cells.

    `cell_corners` has shape (N, 4): columns = [xlow, xhigh, ylow, yhigh].
    Returns K_all (N, 4, 4) complex, K_norm_all (N, 4, 4) real.
    """
    N = cell_corners.shape[0]
    K_all = np.empty((N, 4, 4), dtype=np.complex128)
    K_norm_all = np.empty((N, 4, 4), dtype=np.float64)
    for k in range(N):
        xlow, xhigh, ylow, yhigh = cell_corners[k]
        K_all[k], K_norm_all[k] = element_matrices(xlow, xhigh, ylow, yhigh, k0)
    return K_all, K_norm_all


def batched_element_loads(cell_corners: np.ndarray, k0: float) -> np.ndarray:
    N = cell_corners.shape[0]
    f_all = np.empty((N, 4), dtype=np.complex128)
    for k in range(N):
        xlow, xhigh, ylow, yhigh = cell_corners[k]
        f_all[k] = element_load(xlow, xhigh, ylow, yhigh, k0)
    return f_all
