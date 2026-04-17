"""Fine-scale Q1 reference solver.

Matches Matlab FEM.m exactly: solve the Q1 Dirichlet problem on an
N_f × N_f grid on [0,1]^2 with the midpoint-rule coefficient a(xc, yc) per
cell and constant RHS equal to -1 (Matlab elementstiff2 loads -h^2/4 per
corner, which is the Q1 projection of f ≡ -1).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import q1
from .mesh import local_to_global_nodes, domain_boundary_nodes


def _cell_centers(Nx: int, Ny: int, x0: float = 0.0, x1: float = 1.0,
                  y0: float = 0.0, y1: float = 1.0):
    hx = (x1 - x0) / Nx
    hy = (y1 - y0) / Ny
    xc = x0 + (np.arange(Nx) + 0.5) * hx
    yc = y0 + (np.arange(Ny) + 0.5) * hy
    XC, YC = np.meshgrid(xc, yc, indexing="xy")  # (Ny, Nx)
    return XC, YC


def _coo_from_element_matrices(G: np.ndarray, K_loc: np.ndarray,
                               n_nodes: int) -> sp.csc_matrix:
    """Scatter per-cell (4,4) matrices into a global sparse matrix via COO."""
    ncells = G.shape[0]
    I = np.broadcast_to(G[:, :, None], (ncells, 4, 4)).reshape(-1)
    J = np.broadcast_to(G[:, None, :], (ncells, 4, 4)).reshape(-1)
    V = K_loc.reshape(-1)
    return sp.coo_matrix((V, (I, J)), shape=(n_nodes, n_nodes)).tocsc()


def assemble_fine(a_fun: Callable, N_f: int, rhs_value: float = -1.0):
    """Assemble the fine-scale Q1 stiffness A, mass M, and load F.

    Returns
    -------
    A : csc_matrix, (N+1)^2 x (N+1)^2  — Q1 stiffness with midpoint-rule a
    M : csc_matrix, (N+1)^2 x (N+1)^2  — Q1 mass (no coefficient)
    F : ndarray,     (N+1)^2           — Q1 load ∫ rhs * φ_i, with rhs ≡ rhs_value
    """
    XC, YC = _cell_centers(N_f, N_f)  # (Ny=N_f, Nx=N_f)
    # Cell-centre coefficient values, flattened row-major to match
    # local_to_global_nodes ordering (j-major).
    a_center = np.asarray(a_fun(XC, YC)).ravel()  # (N_f^2,)

    C = q1.C_STIFF  # (4,4), size-independent
    M_ref = q1.C_MASS  # (4,4), needs h^2 factor
    h = 1.0 / N_f
    K_loc = a_center[:, None, None] * C[None, :, :]  # (N_f^2, 4, 4)
    M_loc = np.broadcast_to(h**2 * M_ref, (N_f**2, 4, 4))

    G = local_to_global_nodes(N_f, N_f)
    n_nodes = (N_f + 1) ** 2
    A = _coo_from_element_matrices(G, K_loc, n_nodes)
    M = _coo_from_element_matrices(G, M_loc, n_nodes)

    # Load: for rhs ≡ rhs_value, ∫ rhs * φ_i over each cell = rhs * h^2/4 per
    # corner. Matlab uses rhs_value = -1, hence each corner gets -h^2/4.
    per_corner = rhs_value * h**2 / 4
    F = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(F, G.ravel(), per_corner)
    return A, M, F


def _apply_dirichlet_zero(A: sp.spmatrix, F: np.ndarray, bdy: np.ndarray):
    """Return (A_bc, F_bc) with Dirichlet-0 enforced: rows/cols of A replaced
    with identity on boundary nodes; F zeroed there."""
    n = A.shape[0]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[bdy] = False
    # Build projector P that extracts interior, then re-embed
    interior = np.where(keep_mask)[0]
    # Slice A to interior and solve small system
    A_ii = A[interior][:, interior]
    F_i = F[interior]
    return A_ii.tocsc(), F_i, interior


def solve_fine(a_fun: Callable, N_f: int, rhs_value: float = -1.0):
    """Solve the fine Q1 reference problem. Returns (u, A, M, F).

    `u` has shape ((N_f+1)^2,) with Dirichlet-0 values on the domain boundary.
    `A`, `M` are the unmodified sparse matrices (suitable for energy/L² norms).
    `F` is the unmodified load vector.
    """
    A, M, F = assemble_fine(a_fun, N_f, rhs_value=rhs_value)
    bdy = domain_boundary_nodes(N_f, N_f)
    A_ii, F_i, interior = _apply_dirichlet_zero(A, F, bdy)
    u_i = spla.spsolve(A_ii, F_i)
    u = np.zeros((N_f + 1) ** 2, dtype=np.float64)
    u[interior] = u_i
    return u, A, M, F
