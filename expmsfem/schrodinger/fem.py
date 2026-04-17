"""Fine-scale Q1 reference eigensolver for the 2D Schrödinger equation

    H ψ = E ψ,   H = -½ Δ + V(x)

on `(0, 1)²` with Dirichlet-zero boundary conditions. The generalised
sparse eigenproblem `H ψ = E M ψ` (M = Q1 mass) is solved with
`scipy.sparse.linalg.eigsh` (shift-invert Lanczos) to extract the lowest
`k` eigenpairs cheaply even when the matrix is several tens of thousands.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..mesh import local_to_global_nodes, domain_boundary_nodes
from ..q1 import C_STIFF, C_MASS


def assemble_hamiltonian_and_mass(V_fun: Callable, N_f: int):
    """Assemble the fine-scale Hamiltonian H and mass matrix M on
    `(0, 1)²` at resolution N_f × N_f. Midpoint-rule quadrature for the
    potential; exact Q1 formulas for the kinetic and mass contributions.

    Returns
    -------
    H : (N_f+1)² × (N_f+1)² csc, real Hermitian.
    M : (N_f+1)² × (N_f+1)² csc, Q1 mass.
    """
    h = 1.0 / N_f
    xc = (np.arange(N_f) + 0.5) * h
    yc = (np.arange(N_f) + 0.5) * h
    XC, YC = np.meshgrid(xc, yc, indexing="xy")
    V_centers = np.asarray(V_fun(XC, YC)).ravel()  # (N_f², ), j-major

    # Per-cell 4×4: H_cell = ½·C_STIFF + V_center·h²·C_MASS
    K_loc = (0.5 * C_STIFF)[None, :, :] \
            + V_centers[:, None, None] * (h ** 2) * C_MASS[None, :, :]
    M_loc = np.broadcast_to(h ** 2 * C_MASS, (N_f ** 2, 4, 4))

    G = local_to_global_nodes(N_f, N_f)
    n = (N_f + 1) ** 2
    I = np.broadcast_to(G[:, :, None], (N_f ** 2, 4, 4)).reshape(-1)
    J = np.broadcast_to(G[:, None, :], (N_f ** 2, 4, 4)).reshape(-1)
    H = sp.coo_matrix((K_loc.reshape(-1), (I, J)), shape=(n, n)).tocsc()
    M = sp.coo_matrix((M_loc.reshape(-1), (I, J)), shape=(n, n)).tocsc()
    return H, M


def _interior_submatrices(H, M, N_f):
    bdy = domain_boundary_nodes(N_f, N_f)
    mask = np.ones(H.shape[0], dtype=bool)
    mask[bdy] = False
    interior = np.where(mask)[0]
    H_int = H[interior][:, interior].tocsc()
    M_int = M[interior][:, interior].tocsc()
    return H_int, M_int, interior


def solve_fine_eigenpairs(V_fun: Callable, N_f: int, k: int = 6,
                          sigma: float | None = 0.0):
    """Compute the lowest `k` eigenpairs of `H ψ = E M ψ` on `[0, 1]²`
    at Q1 resolution `N_f × N_f` with Dirichlet-0 BC.

    `sigma` is the shift for the shift-invert Lanczos iteration
    (`scipy.sparse.linalg.eigsh`). Choose it slightly below the expected
    smallest eigenvalue for best convergence; default `0.0` works well for
    non-negative potentials.

    Returns
    -------
    E : (k,) sorted ascending
    psi : (N_f+1)^2 × k real, each column a normalised eigenfunction on the
          full-grid DOFs (Dirichlet boundary values = 0).
    H, M : the sparse matrices (kept for error-norm computations).
    """
    H, M = assemble_hamiltonian_and_mass(V_fun, N_f)
    H_int, M_int, interior = _interior_submatrices(H, M, N_f)
    E, psi_int = spla.eigsh(H_int, k=k, M=M_int, sigma=sigma, which="LM")
    order = np.argsort(E)
    E = E[order]
    psi_int = psi_int[:, order]
    # M-normalise: ψ' M ψ = 1
    for j in range(k):
        nrm = np.sqrt(psi_int[:, j] @ (M_int @ psi_int[:, j]))
        if nrm > 0:
            psi_int[:, j] /= nrm
    # Embed into full grid with zero Dirichlet values
    psi = np.zeros(((N_f + 1) ** 2, k), dtype=np.float64)
    psi[interior, :] = psi_int
    return E, psi, H, M
