"""Reconstruct the fine-mesh multiscale solution from global coefficients and
per-cell basis arrays, adding the local bubble per cell."""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import local_ops
from .assembly import cell_global_indices


def reconstruct_fine(values: np.ndarray, u_global: np.ndarray,
                     bubbles: np.ndarray, N_c: int, N_f: int, N_e: int
                     ) -> np.ndarray:
    """Paste each cell's (basis · coarse_coefficients) + bubble into the
    global (N_c*N_f+1)^2 fine array.

    Parameters
    ----------
    values : (N_c*N_c, (N_f+1)^2, count)   per-cell fine-basis matrices
    u_global : (n_total_dofs,)             global coarse solution
    bubbles : (N_c*N_c, (N_f+1)^2)          per-cell bubble solutions
    """
    N = N_c * N_f
    fine = np.zeros((N + 1, N + 1), dtype=np.float64)
    for n in range(N_c):
        for m in range(N_c):
            cell = n * N_c + m
            gidx = cell_global_indices(m, n, N_c, N_e)
            zs = values[cell] @ u_global[gidx]   # shape ((N_f+1)^2,)
            zs = zs + bubbles[cell]
            zs_2d = zs.reshape(N_f + 1, N_f + 1)
            # Paste into fine grid: cell (m,n) occupies [m*N_f..m*N_f+N_f] in x,
            # [n*N_f..n*N_f+N_f] in y. Local layout is row-major j-then-i, so
            # zs_2d[j, i] is node (i, j) local. Fine array index is [j_global, i_global].
            fine[n * N_f : n * N_f + N_f + 1,
                 m * N_f : m * N_f + N_f + 1] = zs_2d
    return fine.ravel()
