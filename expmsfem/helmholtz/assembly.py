"""Global DOF layout, sparse assembly, and solve for Helmholtz ExpMsFEM.

Reuses the DOF-indexing conventions from `expmsfem.assembly`, but with two
differences that mirror Matlab helmholtz/case1/Exp/MsFEM.m:

  * Only the FOUR DOMAIN CORNERS are Dirichlet-pinned among nodal DOFs
    (the non-corner ∂Ω nodal DOFs stay free — their impedance BC is baked
    into the element stiffness).
  * Boundary-edge auxiliary DOFs (horizontal at k=0 or k=N_c; vertical at
    m=0 or m=N_c) are Dirichlet-pinned to zero (the corresponding basis
    columns were set to zero in element_basis).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..assembly import (
    cell_global_indices,
    all_cell_global_indices,
    n_total,
    n_nodal,
    n_horizontal,
    _h_edge_base,
    _v_edge_base,
)
from ..mesh import node_index


def domain_boundary_dofs_helm(N_c: int, N_e: int) -> np.ndarray:
    """Dirichlet-pinned DOFs for the Helmholtz global system: 4 domain corners
    + all boundary-edge auxiliary DOFs."""
    N_p = N_e + 1
    nn = n_nodal(N_c)
    nh = n_horizontal(N_c, N_e)
    bdy = []
    # 4 corners of Ω
    for (i, j) in [(0, 0), (N_c, 0), (0, N_c), (N_c, N_c)]:
        bdy.append(node_index(N_c, i, j))
    # Horizontal edges at k=0 and k=N_c
    for m in range(N_c):
        base0 = nn + _h_edge_base(m, 0, N_c, N_p)
        bdy.extend(range(base0, base0 + N_p))
        base1 = nn + _h_edge_base(m, N_c, N_c, N_p)
        bdy.extend(range(base1, base1 + N_p))
    # Vertical edges at m=0 and m=N_c
    for n in range(N_c):
        base0 = nn + nh + _v_edge_base(0, n, N_c, N_p)
        bdy.extend(range(base0, base0 + N_p))
        base1 = nn + nh + _v_edge_base(N_c, n, N_c, N_p)
        bdy.extend(range(base1, base1 + N_p))
    return np.array(sorted(set(bdy)), dtype=np.int64)


def assemble_global(K_cells: np.ndarray, f_cells: np.ndarray,
                    cell_idx: np.ndarray, N: int):
    """Complex sparse assembly. K_cells shape (n_cells, count, count)."""
    n_cells, count, _ = K_cells.shape
    I = np.broadcast_to(cell_idx[:, :, None], (n_cells, count, count)).reshape(-1)
    J = np.broadcast_to(cell_idx[:, None, :], (n_cells, count, count)).reshape(-1)
    V = K_cells.reshape(-1)
    A = sp.coo_matrix((V, (I, J)), shape=(N, N)).tocsc()
    F = np.zeros(N, dtype=np.complex128)
    np.add.at(F, cell_idx.ravel(), f_cells.ravel())
    return A, F


def solve_coarse(A: sp.csc_matrix, F: np.ndarray, bdy: np.ndarray):
    """Solve A.T u = F with u[bdy] = 0 (Matlab uses `A.' \\ F`)."""
    n = A.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[bdy] = False
    interior = np.where(mask)[0]
    A_ii = A[interior][:, interior].tocsc()
    F_i = F[interior]
    # Matlab solves `A.' \ F` which for the Helmholtz global is the non-
    # conjugate transpose.
    u_i = spla.spsolve(A_ii.T.tocsc(), F_i)
    u = np.zeros(n, dtype=np.complex128)
    u[interior] = u_i
    return u
