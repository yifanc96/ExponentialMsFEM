"""Fine-scale Q1 reference solver for the Helmholtz impedance problem.

Mirrors Matlab helmholtz/case1/Exp/FEM.m: no Dirichlet elimination (impedance
BC is fully baked into the stiffness matrix via the β terms in elementstiff1),
solves the transpose system `A.T u = F`, returns (u, A=K_helm, B=K_norm,
C=Q1_mass).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..mesh import local_to_global_nodes
from ..q1 import C_MASS
from . import q1 as hq1


def _cell_corners_grid(N_f: int, x0: float = 0.0, x1: float = 1.0,
                       y0: float = 0.0, y1: float = 1.0) -> np.ndarray:
    """Return shape (N_f*N_f, 4) = [xlow, xhigh, ylow, yhigh] per cell,
    in the j-major (row-by-row) cell ordering used throughout the codebase.
    """
    xs = np.linspace(x0, x1, N_f + 1)
    ys = np.linspace(y0, y1, N_f + 1)
    rows = []
    for j in range(N_f):
        for i in range(N_f):
            rows.append([xs[i], xs[i + 1], ys[j], ys[j + 1]])
    return np.array(rows)


def assemble_fine(N_f: int, k0: float):
    """Assemble the fine-scale complex stiffness A, real norm-matrix B, Q1
    mass C, and load vector F on [0, 1]²."""
    cell_corners = _cell_corners_grid(N_f)
    K_loc, Knorm_loc = hq1.batched_element_matrices(cell_corners, k0)
    f_loc = hq1.batched_element_loads(cell_corners, k0)

    G = local_to_global_nodes(N_f, N_f)
    n_nodes = (N_f + 1) ** 2
    I = np.broadcast_to(G[:, :, None], (N_f ** 2, 4, 4)).reshape(-1)
    J = np.broadcast_to(G[:, None, :], (N_f ** 2, 4, 4)).reshape(-1)

    A = sp.coo_matrix((K_loc.reshape(-1), (I, J)), shape=(n_nodes, n_nodes)).tocsc()
    B = sp.coo_matrix((Knorm_loc.reshape(-1), (I, J)),
                      shape=(n_nodes, n_nodes)).tocsc()

    h = 1.0 / N_f
    M_loc = np.broadcast_to((h ** 2) * C_MASS, (N_f ** 2, 4, 4))
    C = sp.coo_matrix((M_loc.reshape(-1), (I, J)),
                      shape=(n_nodes, n_nodes)).tocsc()

    F = np.zeros(n_nodes, dtype=np.complex128)
    np.add.at(F, G.ravel(), f_loc.reshape(-1))
    return A, B, C, F


def solve_fine(N_f: int, k0: float):
    """Solve the fine-scale Helmholtz impedance system. Returns
    (u complex, B, C, F)."""
    A, B, C, F = assemble_fine(N_f, k0)
    # Matlab uses `A.' \ F` (non-conjugate transpose). Translating to scipy:
    u = spla.spsolve(A.T.tocsc(), F)
    return u, B, C, F
