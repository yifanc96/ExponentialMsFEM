"""Per-cell ExpMsFEM basis for the Schrödinger eigenvalue problem.

Differences from the elliptic element basis:

  * No RHS-driven bubble — the eigenvalue problem is homogeneous.
  * **Cell-interior modes.** Because the eigenfunctions can peak inside
    a coarse cell (think: ground-state Gaussian of a harmonic well whose
    maximum is strictly interior to cell `(m, n)`), we add the first
    `K_int` Dirichlet-in-cell eigenfunctions of the local Hamiltonian
    `H_cell` (M-orthonormalised, zero on the cell perimeter) as extra
    basis columns. These are cell-unique (not shared) — they live at
    the end of the global DOF vector. Each cell contributes

        count = 4   (nodal hats)
              + 4 · N_e   (edge eigen-modes, 4 edges)
              + K_int   (cell-interior Dirichlet eigenfunctions)

  * The coarse online phase assembles `H_c` and `M_c` from these
    per-cell blocks and solves the small dense generalised eigenproblem
    `H_c c = E M_c c`.

Without the `K_int` cell-interior modes, the basis is boundary-supported
and cannot reproduce eigenfunctions whose peaks sit strictly inside a
cell — the eigenvalue error then plateaus regardless of `N_e`.
"""

from __future__ import annotations

import concurrent.futures as cf

import numpy as np
import scipy.linalg as sla

from ..mesh import cell_boundary_nodes
from . import local_ops


def _eigen_modes(R: np.ndarray, N: np.ndarray, P: np.ndarray, N_e: int):
    """Top-N_e generalised eigenvectors of `R^T N R v = λ P v`, ranked by |λ|.
    R^T N R has rank ≤ `R.shape[0]`. For positive-definite `V(x)` both `N`
    and `P` are SPD and we use the fast `eigh` + `subset_by_index` path;
    for indefinite Hamiltonians (deep attractive potentials, lattices) `P`
    can become indefinite and we fall back to the general `eig` solver,
    sorted by |λ|."""
    M = R.T @ N @ R
    M = 0.5 * (M + M.T)
    P_reg = P + 1e-14 * np.eye(P.shape[0])
    P_reg = 0.5 * (P_reg + P_reg.T)
    n = M.shape[0]
    try:
        lo = max(0, n - N_e)
        _, eigvecs = sla.eigh(M, P_reg, subset_by_index=[lo, n - 1])
        return eigvecs[:, ::-1]
    except np.linalg.LinAlgError:
        # Indefinite P — use the dense general solver
        eigvals, eigvecs = sla.eig(M, P_reg)
        order = np.argsort(-np.abs(eigvals))
        return np.real_if_close(eigvecs[:, order[:N_e]], tol=1e6)


def _nodal_basis_rhs(N_f: int) -> np.ndarray:
    n_per = 4 * N_f
    f = np.zeros((n_per, 4), dtype=np.float64)
    f[0:N_f + 1, 0] = np.linspace(1, 0, N_f + 1)
    f[N_f + 1:2 * N_f + 1, 0] = np.linspace(1 - 1 / N_f, 0, N_f)
    f[0:N_f + 1, 1] = np.linspace(0, 1, N_f + 1)
    f[2 * N_f + 1:3 * N_f + 1, 1] = np.linspace(1 - 1 / N_f, 0, N_f)
    f[2 * N_f + 1:3 * N_f + 1, 2] = np.linspace(1 / N_f, 1, N_f)
    f[3 * N_f + 1:4 * N_f, 2] = np.linspace(1 / N_f, 1 - 1 / N_f, N_f - 1)
    f[N_f + 1:2 * N_f + 1, 3] = np.linspace(1 / N_f, 1, N_f)
    f[3 * N_f + 1:4 * N_f, 3] = np.linspace(1 - 1 / N_f, 1 / N_f, N_f - 1)
    return f


def _nodal_basis(ws: local_ops.Workspace, m: int, n: int) -> np.ndarray:
    """The 4 corner-hat nodal basis functions — `H`-harmonic extensions of
    piecewise-linear perimeter data. Same construction as the elliptic path
    but with the Hamiltonian standing in as the cell operator."""
    H_cell, _, bdy, lu = ws.cell_factor(m, n)
    f = _nodal_basis_rhs(ws.N_f)
    n_nodes = (ws.N_f + 1) ** 2
    F_zero = np.zeros((n_nodes, 4))
    return lu.solve(F_zero, bdy, f)


def _build_edge_data(ws: local_ops.Workspace, t, m_edge, n_edge, N_e):
    L1, L2, N_mat = local_ops.harmext(ws, m_edge, n_edge, t)
    R, P = local_ops.restrict(ws, m_edge, n_edge, t)
    V = _eigen_modes(R, N_mat, P, N_e)
    RV = R @ V
    return (L1 @ RV, L2 @ RV)


def prefactor_edges(ws: local_ops.Workspace, N_e: int,
                    n_workers: int | None = None):
    """Populate `ws._edge_cache` for every interior edge in parallel."""
    ws._edge_cache = {}
    N_c = ws.N_c
    keys = ([(1, m, n) for n in range(N_c - 1) for m in range(N_c)]
            + [(2, m, n) for n in range(N_c) for m in range(N_c - 1)])

    def _one(key):
        t, m, n = key
        return key, _build_edge_data(ws, t, m, n, N_e)

    with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for key, val in pool.map(_one, keys):
            ws._edge_cache[key] = val


def _edge_contribution(ws, m_edge, n_edge, t, side, N_e):
    L1_RV, L2_RV = ws._edge_cache[(t, m_edge, n_edge)]
    return L1_RV if side == "low" else L2_RV


def _cell_interior_modes(H_cell, M_cell, bdy, K_int: int):
    """First `K_int` Dirichlet-in-cell generalised eigenfunctions of the
    local Hamiltonian, M-normalised and embedded into the full-cell DOF
    vector (zero on the cell perimeter)."""
    n = H_cell.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[bdy] = False
    interior = np.where(mask)[0]
    H_int = H_cell[interior][:, interior].toarray()
    M_int = M_cell[interior][:, interior].toarray()
    H_int = 0.5 * (H_int + H_int.T)
    M_int = 0.5 * (M_int + M_int.T)
    _, psi = sla.eigh(H_int, M_int, subset_by_index=[0, K_int - 1])
    # M-normalise
    for j in range(K_int):
        nrm = np.sqrt(psi[:, j] @ (M_int @ psi[:, j]))
        if nrm > 0:
            psi[:, j] /= nrm
    full = np.zeros((n, K_int), dtype=np.float64)
    full[interior, :] = psi
    return full


def element_basis(ws: local_ops.Workspace, m: int, n: int,
                  N_e: int, K_int: int):
    """Full per-cell Schrödinger basis + coarse Hamiltonian and mass blocks.

    Returns `(value, H_elem, M_elem)` with `value` shape `((N_f+1)², count)`
    and `count = 4 + 4·N_e + K_int`. The final `K_int` columns are the
    Dirichlet-in-cell Hamiltonian eigenfunctions (zero on the cell
    perimeter), which provide cell-interior representation.
    """
    N_c, N_f = ws.N_c, ws.N_f
    count = 4 + 4 * N_e + K_int
    n_nodes = (N_f + 1) ** 2
    value = np.zeros((n_nodes, count), dtype=np.float64)

    H_cell, M_cell, bdy, _ = ws.cell_factor(m, n)
    value[:, 0:4] = _nodal_basis(ws, m, n)

    off = 4
    if n > 0:
        value[:, off:off + N_e] = _edge_contribution(ws, m, n - 1, 1, "high", N_e)
    off += N_e
    if n < N_c - 1:
        value[:, off:off + N_e] = _edge_contribution(ws, m, n, 1, "low", N_e)
    off += N_e
    if m > 0:
        value[:, off:off + N_e] = _edge_contribution(ws, m - 1, n, 2, "high", N_e)
    off += N_e
    if m < N_c - 1:
        value[:, off:off + N_e] = _edge_contribution(ws, m, n, 2, "low", N_e)
    off += N_e

    if K_int > 0:
        value[:, off:off + K_int] = _cell_interior_modes(H_cell, M_cell, bdy,
                                                         K_int)

    H_elem = value.T @ (H_cell @ value)
    M_elem = value.T @ (M_cell @ value)
    return value, H_elem, M_elem
