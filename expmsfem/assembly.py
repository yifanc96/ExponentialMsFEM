"""Global DOF layout, sparse assembly, Dirichlet elimination, coarse solve,
and fine-scale reconstruction for ExpMsFEM. Mirrors Matlab MsFEM.m global
indexing in 0-indexed form.

DOF layout (total = n_nodal + n_h_edge_dof + n_v_edge_dof):

  Nodal corners    [0, n_nodal)                   n_nodal = (N_c+1)^2
    Node (i, j) = j*(N_c+1) + i for i,j in 0..N_c

  Horizontal edges [n_nodal, n_nodal + n_h)       n_h = N_c * (N_c+1) * N_p
    Edge at (x-col m, y-row k) with m ∈ 0..N_c-1, k ∈ 0..N_c.
    Block base = n_nodal + (m + k*N_c) * N_p; slot ∈ 0..N_p-1.

  Vertical edges   [n_nodal + n_h, n_nodal + n_h + n_v)   n_v = (N_c+1) * N_c * N_p
    Edge at (x-col m, y-row n) with m ∈ 0..N_c, n ∈ 0..N_c-1.
    Block base = n_nodal + n_h + (m + n*(N_c+1)) * N_p; slot ∈ 0..N_p-1.

For cell (m, n) with 0-indexed m, n ∈ 0..N_c-1 and local slots in order
[4 corners | bottom edge N_p | top edge N_p | left edge N_p | right edge N_p]
with N_p = N_e + 1:

  local 0 → node (m,   n)                 (bottom-left corner)
  local 1 → node (m+1, n)                 (bottom-right)
  local 2 → node (m+1, n+1)               (top-right)
  local 3 → node (m,   n+1)               (top-left)
  local 4+slot, slot ∈ 0..N_p-1 → bottom edge: h-edge at (m, n)
  local 4+N_p+slot                       → top edge:    h-edge at (m, n+1)
  local 4+2 N_p+slot                     → left edge:   v-edge at (m, n)
  local 4+3 N_p+slot                     → right edge:  v-edge at (m+1, n)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import node_index


def n_nodal(N_c: int) -> int:
    return (N_c + 1) ** 2


def n_horizontal(N_c: int, N_e: int, n_per_edge: int | None = None) -> int:
    """DOF count for all horizontal edges. `n_per_edge` defaults to N_e+1 for
    the Exp method (N_e eigenmodes + 1 edge-bubble per edge); baselines pass
    `n_per_edge=N_e` for H+bubble / O(H)."""
    if n_per_edge is None:
        n_per_edge = N_e + 1
    return N_c * (N_c + 1) * n_per_edge


def n_vertical(N_c: int, N_e: int, n_per_edge: int | None = None) -> int:
    if n_per_edge is None:
        n_per_edge = N_e + 1
    return (N_c + 1) * N_c * n_per_edge


def n_total(N_c: int, N_e: int, n_per_edge: int | None = None) -> int:
    return (n_nodal(N_c)
            + n_horizontal(N_c, N_e, n_per_edge)
            + n_vertical(N_c, N_e, n_per_edge))


def _h_edge_base(m: int, k: int, N_c: int, N_p: int) -> int:
    return (m + k * N_c) * N_p


def _v_edge_base(m: int, n: int, N_c: int, N_p: int) -> int:
    return (m + n * (N_c + 1)) * N_p


def cell_global_indices(m: int, n: int, N_c: int, N_e: int,
                        n_per_edge: int | None = None) -> np.ndarray:
    """Global DOF indices for each local slot of cell (m, n).

    Returns int array of shape (count,) with count = 4 + 4 * n_per_edge
    (default n_per_edge = N_e + 1 for the Exp method).
    """
    if n_per_edge is None:
        n_per_edge = N_e + 1
    N_p = n_per_edge
    count = 4 + 4 * N_p
    out = np.empty(count, dtype=np.int64)

    # 4 corners
    out[0] = node_index(N_c, m, n)
    out[1] = node_index(N_c, m + 1, n)
    out[2] = node_index(N_c, m + 1, n + 1)
    out[3] = node_index(N_c, m, n + 1)

    nn = n_nodal(N_c)
    nh = n_horizontal(N_c, N_e, n_per_edge)

    # Bottom edge (horizontal at k=n, x-col m): slots 4..4+N_p
    hbase_bot = nn + _h_edge_base(m, n, N_c, N_p)
    out[4 : 4 + N_p] = hbase_bot + np.arange(N_p)
    # Top edge (horizontal at k=n+1, x-col m)
    hbase_top = nn + _h_edge_base(m, n + 1, N_c, N_p)
    out[4 + N_p : 4 + 2 * N_p] = hbase_top + np.arange(N_p)
    # Left edge (vertical at x-col m, y-row n)
    vbase_left = nn + nh + _v_edge_base(m, n, N_c, N_p)
    out[4 + 2 * N_p : 4 + 3 * N_p] = vbase_left + np.arange(N_p)
    # Right edge (vertical at x-col m+1, y-row n)
    vbase_right = nn + nh + _v_edge_base(m + 1, n, N_c, N_p)
    out[4 + 3 * N_p : 4 + 4 * N_p] = vbase_right + np.arange(N_p)

    return out


def all_cell_global_indices(N_c: int, N_e: int,
                            n_per_edge: int | None = None) -> np.ndarray:
    """Stacked global DOF indices per cell, shape (N_c*N_c, count)."""
    if n_per_edge is None:
        n_per_edge = N_e + 1
    count = 4 + 4 * n_per_edge
    out = np.empty((N_c * N_c, count), dtype=np.int64)
    for n in range(N_c):
        for m in range(N_c):
            cell_idx = n * N_c + m
            out[cell_idx] = cell_global_indices(m, n, N_c, N_e, n_per_edge)
    return out


def domain_boundary_dofs(N_c: int, N_e: int,
                         n_per_edge: int | None = None) -> np.ndarray:
    """All DOFs that must be zero because they live on ∂Ω:
       - nodal corners with i ∈ {0, N_c} or j ∈ {0, N_c}
       - horizontal edges at k = 0 or k = N_c
       - vertical edges at m = 0 or m = N_c
    """
    if n_per_edge is None:
        n_per_edge = N_e + 1
    N_p = n_per_edge
    nn = n_nodal(N_c)
    nh = n_horizontal(N_c, N_e, n_per_edge)

    bdy = []

    # Nodal corners on ∂Ω
    for i in range(N_c + 1):
        for j in range(N_c + 1):
            if i == 0 or j == 0 or i == N_c or j == N_c:
                bdy.append(node_index(N_c, i, j))

    # Horizontal edges at k=0 (bottom of Ω)
    for m in range(N_c):
        base = nn + _h_edge_base(m, 0, N_c, N_p)
        bdy.extend(range(base, base + N_p))
    # Horizontal edges at k=N_c (top)
    for m in range(N_c):
        base = nn + _h_edge_base(m, N_c, N_c, N_p)
        bdy.extend(range(base, base + N_p))

    # Vertical edges at m=0 (left of Ω)
    for n in range(N_c):
        base = nn + nh + _v_edge_base(0, n, N_c, N_p)
        bdy.extend(range(base, base + N_p))
    # Vertical edges at m=N_c (right)
    for n in range(N_c):
        base = nn + nh + _v_edge_base(N_c, n, N_c, N_p)
        bdy.extend(range(base, base + N_p))

    return np.array(sorted(set(bdy)), dtype=np.int64)


def assemble_global(K_cells: np.ndarray, f_cells: np.ndarray,
                    cell_idx: np.ndarray, N: int):
    """Assemble sparse global matrix and load.

    Parameters
    ----------
    K_cells : (n_cells, count, count)  element stiffness per cell (Fortran flatten)
    f_cells : (n_cells, count)         element load per cell
    cell_idx : (n_cells, count)        global DOF indices per local slot
    N : int  total global DOF count

    Returns (A csc, F ndarray).
    """
    n_cells, count, _ = K_cells.shape
    # COO triplets
    I = np.broadcast_to(cell_idx[:, :, None], (n_cells, count, count)).reshape(-1)
    J = np.broadcast_to(cell_idx[:, None, :], (n_cells, count, count)).reshape(-1)
    V = K_cells.reshape(-1)
    A = sp.coo_matrix((V, (I, J)), shape=(N, N)).tocsc()
    F = np.zeros(N, dtype=np.float64)
    np.add.at(F, cell_idx.ravel(), f_cells.ravel())
    return A, F


def solve_coarse(A: sp.csc_matrix, F: np.ndarray, bdy: np.ndarray):
    """Solve A u = F with Dirichlet-0 on `bdy`. Uses interior-only sparse solve."""
    n = A.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[bdy] = False
    interior = np.where(mask)[0]
    A_ii = A[interior][:, interior].tocsc()
    F_i = F[interior]
    u_i = spla.spsolve(A_ii, F_i)
    u = np.zeros(n, dtype=np.float64)
    u[interior] = u_i
    return u
