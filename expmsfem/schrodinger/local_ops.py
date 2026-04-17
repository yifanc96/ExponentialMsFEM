"""Per-cell / per-patch operators for the Schrödinger ExpMsFEM.

Structurally the same as the elliptic pipeline, but the per-cell "stiffness"
is the full Hamiltonian `H = -½ Δ + V(x)` assembled via midpoint quadrature
for the potential. Boundary conditions on each local problem are Dirichlet-0
on the cell / patch perimeter (identical to the elliptic case — the
physically interesting boundary condition on `∂Ω` is also Dirichlet-0).

Workspace also caches the per-cell mass matrix `M_cell` alongside the
Hamiltonian factor; both are needed later to build the coarse generalised
eigenproblem.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..mesh import cell_boundary_nodes, local_to_global_nodes
from ..q1 import C_STIFF, C_MASS


def _assemble_Q1_hamiltonian(V_fun: Callable,
                             xs: np.ndarray, ys: np.ndarray):
    """Return `(H_local, M_local)` on the (Nx+1)×(Ny+1) fine grid defined
    by `xs, ys`. `H = ½·kinetic + V·mass` cell-by-cell via midpoint rule."""
    Nx = xs.size - 1
    Ny = ys.size - 1
    # Fine-cell midpoints
    xmid = 0.5 * (xs[:-1] + xs[1:])
    ymid = 0.5 * (ys[:-1] + ys[1:])
    XC, YC = np.meshgrid(xmid, ymid, indexing="xy")
    V_c = np.asarray(V_fun(XC, YC)).ravel()

    # Non-square cells possible for oversampled patches, but in our
    # framework they're squares of side `h = xs[1]-xs[0]`.
    h = xs[1] - xs[0]
    h2 = h * h
    # Q1 stiffness is scale-free on squares, mass scales with h².
    K_loc = (0.5 * C_STIFF)[None, :, :] + V_c[:, None, None] * h2 * C_MASS[None, :, :]
    M_loc = np.broadcast_to(h2 * C_MASS, (Nx * Ny, 4, 4))

    G = local_to_global_nodes(Nx, Ny)
    n = (Nx + 1) * (Ny + 1)
    I = np.broadcast_to(G[:, :, None], (Nx * Ny, 4, 4)).reshape(-1)
    J = np.broadcast_to(G[:, None, :], (Nx * Ny, 4, 4)).reshape(-1)
    H = sp.coo_matrix((K_loc.reshape(-1), (I, J)), shape=(n, n)).tocsc()
    M = sp.coo_matrix((M_loc.reshape(-1), (I, J)), shape=(n, n)).tocsc()
    return H, M


class InteriorFactor:
    """LU factor of `A[interior, interior]` that supports fast Dirichlet
    solves with arbitrary boundary data."""

    __slots__ = ("n", "interior", "A_ib", "lu")

    def __init__(self, A: sp.csc_matrix, bdy: np.ndarray):
        n = A.shape[0]
        mask = np.ones(n, dtype=bool)
        mask[bdy] = False
        self.n = n
        self.interior = np.where(mask)[0]
        A_csr = A.tocsr()
        A_ii = A_csr[self.interior][:, self.interior].tocsc()
        self.A_ib = A_csr[self.interior][:, bdy]
        self.lu = spla.splu(A_ii)

    def solve(self, F: np.ndarray, bdy: np.ndarray,
              bdy_values: np.ndarray) -> np.ndarray:
        rhs = F[self.interior] - self.A_ib @ bdy_values
        u_i = self.lu.solve(np.asarray(rhs, dtype=np.float64))
        out_shape = (self.n,) if rhs.ndim == 1 else (self.n, rhs.shape[1])
        u = np.zeros(out_shape, dtype=np.float64)
        u[self.interior] = u_i
        u[bdy] = bdy_values
        return u


def basefun(V_fun: Callable, m: int, n: int, N_c: int, N_f: int):
    """Hamiltonian H_cell and mass M_cell on coarse cell (m, n)."""
    H = 1.0 / N_c
    xs = np.linspace(m * H, (m + 1) * H, N_f + 1)
    ys = np.linspace(n * H, (n + 1) * H, N_f + 1)
    return _assemble_Q1_hamiltonian(V_fun, xs, ys)


def _patch_bounds(m: int, n: int, N_c: int, t: int):
    """Oversampled-patch range — same 3-wide × 2-wide geometry as elliptic."""
    if t == 1:
        if m == 0:
            m0, Nx_cells = 0, 2
        elif m == N_c - 1:
            m0, Nx_cells = N_c - 2, 2
        else:
            m0, Nx_cells = m - 1, 3
        n0, Ny_cells = n, 2
    else:
        if n == 0:
            n0, Ny_cells = 0, 2
        elif n == N_c - 1:
            n0, Ny_cells = N_c - 2, 2
        else:
            n0, Ny_cells = n - 1, 3
        m0, Nx_cells = m, 2
    return m0, Nx_cells, n0, Ny_cells


def basefun1(V_fun: Callable, m: int, n: int, N_c: int, N_f: int, t: int):
    """Oversampled-patch H and M matrices, plus patch sizes and offsets."""
    m0, Nx_cells, n0, Ny_cells = _patch_bounds(m, n, N_c, t)
    H = 1.0 / N_c
    xs = np.linspace(m0 * H, (m0 + Nx_cells) * H, Nx_cells * N_f + 1)
    ys = np.linspace(n0 * H, (n0 + Ny_cells) * H, Ny_cells * N_f + 1)
    Hmat, Mmat = _assemble_Q1_hamiltonian(V_fun, xs, ys)
    return Hmat, Mmat, xs.size - 1, ys.size - 1, m0, n0


class Workspace:
    """Caches per-cell and per-patch factors for one Schrödinger ExpMsFEM run."""

    def __init__(self, V_fun: Callable, N_c: int, N_f: int):
        self.V_fun = V_fun
        self.N_c = N_c
        self.N_f = N_f
        self._cell = {}          # (m, n) → (H_cell, M_cell, bdy, InteriorFactor)
        self._patch = {}         # (m, n, t) → (H_patch, M_patch, Nx, Ny, m0, n0, bdy, factor)
        self._edge_cache = {}    # (t, m_edge, n_edge) → (L1·R·V_sel, L2·R·V_sel)

    def _build_cell(self, m: int, n: int):
        H, M = basefun(self.V_fun, m, n, self.N_c, self.N_f)
        bdy = cell_boundary_nodes(self.N_f, self.N_f)
        return (H, M, bdy, InteriorFactor(H, bdy))

    def _build_patch(self, m: int, n: int, t: int):
        Hmat, Mmat, Nx, Ny, m0, n0 = basefun1(self.V_fun, m, n,
                                               self.N_c, self.N_f, t)
        bdy = cell_boundary_nodes(Nx, Ny)
        return (Hmat, Mmat, Nx, Ny, m0, n0, bdy, InteriorFactor(Hmat, bdy))

    def cell_factor(self, m, n):
        k = (m, n)
        v = self._cell.get(k)
        if v is None:
            v = self._build_cell(m, n)
            self._cell[k] = v
        return v

    def patch_factor(self, m, n, t):
        k = (m, n, t)
        v = self._patch.get(k)
        if v is None:
            v = self._build_patch(m, n, t)
            self._patch[k] = v
        return v

    def prefactor_all(self, n_workers: int | None = None):
        import concurrent.futures as cf

        N_c = self.N_c
        cell_keys = [(m, n) for n in range(N_c) for m in range(N_c)]
        patch_keys = ([(m, n, 1) for n in range(N_c - 1) for m in range(N_c)]
                      + [(m, n, 2) for n in range(N_c) for m in range(N_c - 1)])
        with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
            cjobs = {pool.submit(self._build_cell, m, n): (m, n)
                     for (m, n) in cell_keys}
            pjobs = {pool.submit(self._build_patch, m, n, t): (m, n, t)
                     for (m, n, t) in patch_keys}
            for fut in cf.as_completed(cjobs):
                self._cell[cjobs[fut]] = fut.result()
            for fut in cf.as_completed(pjobs):
                self._patch[pjobs[fut]] = fut.result()


# -----------------------------------------------------------------------------
# harmext / restrict (identical structure to elliptic — reuses the cached
# Hamiltonian instead of the elliptic stiffness).
# -----------------------------------------------------------------------------


def _harmext_rhs_matrices(N_f: int, i: int):
    def identity_on(segment: slice):
        f = np.zeros((4 * N_f, N_f - 1), dtype=np.float64)
        idx = np.arange(4 * N_f)[segment]
        f[idx, np.arange(N_f - 1)] = 1.0
        return f

    if i == 1:
        return (identity_on(slice(3 * N_f + 1, 4 * N_f)),
                identity_on(slice(1, N_f)))
    return (identity_on(slice(2 * N_f + 1, 3 * N_f)),
            identity_on(slice(N_f + 1, 2 * N_f)))


def harmext(ws: Workspace, m: int, n: int, i: int):
    """Hamiltonian-harmonic extension of identity edge-hat data on the shared
    edge between (m, n) and its neighbour in direction i."""
    H1, _, b1, lu1 = ws.cell_factor(m, n)
    other = (m, n + 1) if i == 1 else (m + 1, n)
    H2, _, b2, lu2 = ws.cell_factor(*other)
    f1, f2 = _harmext_rhs_matrices(ws.N_f, i)
    n_nodes = (ws.N_f + 1) ** 2
    F_zero = np.zeros((n_nodes, ws.N_f - 1))
    L1 = lu1.solve(F_zero, b1, f1)
    L2 = lu2.solve(F_zero, b2, f2)
    N_mat = L1.T @ (H1 @ L1) + L2.T @ (H2 @ L2)
    return L1, L2, N_mat


def _restrict_active_mask(N_c: int, N_f: int, m: int, n: int, t: int,
                          N_x: int, N_y: int) -> np.ndarray:
    """Perimeter mask of active Dirichlet DOFs — same ladder as elliptic."""
    P_size = 2 * (N_x + N_y)
    mask = np.ones(P_size, dtype=bool)

    def zero_range_1idx(a, b):
        mask[a - 1 : b] = False

    M, Nn = m + 1, n + 1
    Nc = N_c
    interior_m = 1 < M < Nc
    interior_n = 1 < Nn < Nc

    if t == 1:
        if Nn == 1 and (2 < M < Nc - 1):
            zero_range_1idx(1, N_x + 1)
        elif Nn == Nc - 1 and (2 < M < Nc - 1):
            zero_range_1idx(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
        elif (M < 3) and interior_n:
            zero_range_1idx(N_x + 2, N_x + N_y + 1)
        elif (M > Nc - 2) and interior_n:
            zero_range_1idx(N_x + N_y + 2, N_x + 2 * N_y + 1)
        elif Nn == 1 and M < 3:
            zero_range_1idx(1, N_x + 1)
            zero_range_1idx(N_x + 2, N_x + N_y + 1)
        elif Nn == 1 and M > Nc - 2:
            zero_range_1idx(1, N_x + 1)
            zero_range_1idx(N_x + N_y + 2, N_x + 2 * N_y + 1)
        elif Nn == Nc - 1 and M < 3:
            zero_range_1idx(N_x + 2, N_x + N_y + 1)
            zero_range_1idx(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
        elif Nn == Nc - 1 and M > Nc - 2:
            zero_range_1idx(N_x + N_y + 2, N_x + 2 * N_y + 1)
            zero_range_1idx(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
    else:
        if M == 1 and (2 < Nn < Nc - 1):
            zero_range_1idx(N_x + 2, N_x + N_y + 1)
        elif M == Nc - 1 and (2 < Nn < Nc - 1):
            zero_range_1idx(N_x + N_y + 2, N_x + 2 * N_y + 1)
        elif (Nn < 3) and interior_m:
            zero_range_1idx(1, N_x + 1)
        elif (Nn > Nc - 2) and interior_m:
            zero_range_1idx(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
        elif M == 1 and Nn < 3:
            zero_range_1idx(N_x + 2, N_x + N_y + 1)
            zero_range_1idx(1, N_x + 1)
        elif M == Nc - 1 and Nn < 3:
            zero_range_1idx(N_x + N_y + 2, N_x + 2 * N_y + 1)
            zero_range_1idx(1, N_x + 1)
        elif M == 1 and Nn > Nc - 2:
            zero_range_1idx(N_x + 2, N_x + N_y + 1)
            zero_range_1idx(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
        elif M == Nc - 1 and Nn > Nc - 2:
            zero_range_1idx(N_x + N_y + 2, N_x + 2 * N_y + 1)
            zero_range_1idx(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
    mask[-1] = False
    return mask


def _edge_trace_indices(t, m, n, N_c, N_f, Nx_fine, Ny_fine):
    if t == 1:
        if m == 0:
            col_start, col_end = 0, N_f + 1
        else:
            col_start, col_end = N_f, 2 * N_f + 1
        row_idx = N_f
        return row_idx * (Nx_fine + 1) + np.arange(col_start, col_end)
    if n == 0:
        row_start, row_end = 0, N_f + 1
    else:
        row_start, row_end = N_f, 2 * N_f + 1
    col_idx = N_f
    return np.arange(row_start, row_end) * (Nx_fine + 1) + col_idx


def restrict(ws: Workspace, m: int, n: int, t: int):
    """Oversampled-patch Hamiltonian-harmonic lift restricted to the shared
    edge. Returns (R, P) — the edge Schrödinger problem has no RHS-driven
    bubble so the third slot of the elliptic `(R, P, bub)` tuple is omitted."""
    Hmat, _, Nx, Ny, m0, n0, b_patch, lu = ws.patch_factor(m, n, t)
    mask = _restrict_active_mask(ws.N_c, ws.N_f, m, n, t, Nx, Ny)
    active_idx = np.where(mask)[0]
    n_active = active_idx.size
    bdy_vals = np.zeros((b_patch.size, n_active))
    bdy_vals[active_idx, np.arange(n_active)] = 1.0

    n_patch = (Nx + 1) * (Ny + 1)
    F_zero = np.zeros((n_patch, n_active))
    harm = lu.solve(F_zero, b_patch, bdy_vals)
    P = harm.T @ (Hmat @ harm)

    edge_nodes = _edge_trace_indices(t, m, n, ws.N_c, ws.N_f, Nx, Ny)
    leng = ws.N_f + 1
    R_full = harm[edge_nodes, :]
    lin_left = np.linspace(1, 0, leng)
    lin_right = np.linspace(0, 1, leng)
    R_full = R_full - np.outer(lin_left, R_full[0, :]) - np.outer(lin_right, R_full[-1, :])
    return R_full[1 : leng - 1, :], P
