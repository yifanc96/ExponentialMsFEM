"""Local (per-coarse-cell and per-patch) operators that build the ExpMsFEM
basis. All routines mirror the Matlab reference in `elliptic/periodic/Exp`:

  basefun   (cell fine-stiffness)   -> local_stiffness
  basefun1  (oversampled patch)     -> patch_stiffness
  bubble    (constant-RHS bubble)   -> cell_bubble
  harmext   (L1, L2, N)             -> harmonic_extension
  restrict  (R, P, bub on edge)     -> oversampled_restrict

We use scipy.sparse for all local solves. Dirichlet-0 boundary conditions are
enforced by restricting to interior DOFs (sliced sparse system) rather than
Matlab's "A(b,:)=0; A(b,b)=I" trick.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import q1
from .mesh import cell_boundary_nodes, local_to_global_nodes


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _assemble_q1_on_grid(a_centers: np.ndarray, Nx: int, Ny: int) -> sp.csc_matrix:
    """Assemble a Q1 stiffness on an (Nx+1) x (Ny+1) grid with per-cell
    coefficient `a_centers` (shape (Nx*Ny,), row-major j-then-i).

    Only the 4x4 reference stiffness C times a(center) is needed — the entries
    are scale-free for square cells, so patch physical size never appears.
    """
    G = local_to_global_nodes(Nx, Ny)
    C = q1.C_STIFF
    K_loc = a_centers[:, None, None] * C[None, :, :]
    n = (Nx + 1) * (Ny + 1)
    I = np.broadcast_to(G[:, :, None], (Nx * Ny, 4, 4)).reshape(-1)
    J = np.broadcast_to(G[:, None, :], (Nx * Ny, 4, 4)).reshape(-1)
    V = K_loc.reshape(-1)
    return sp.coo_matrix((V, (I, J)), shape=(n, n)).tocsc()


def _cell_center_coords(m: int, n: int, N_c: int, N_f: int):
    """Fine element centre (x,y) coordinates inside coarse cell (m,n)
    (0-indexed), arranged row-major j-then-i. Returns shape (N_f*N_f,)."""
    H = 1.0 / N_c
    h = H / N_f
    xi = m * H + (np.arange(N_f) + 0.5) * h
    yj = n * H + (np.arange(N_f) + 0.5) * h
    XC, YC = np.meshgrid(xi, yj, indexing="xy")  # (Ny=N_f, Nx=N_f)
    return XC.ravel(), YC.ravel()


def _patch_center_coords(m0: int, n0: int, Nx_cells: int, Ny_cells: int,
                         N_c: int, N_f: int):
    """Fine element centres for an oversampled patch that spans coarse cells
    [m0..m0+Nx_cells-1] x [n0..n0+Ny_cells-1]."""
    H = 1.0 / N_c
    h = H / N_f
    Nx_fine = Nx_cells * N_f
    Ny_fine = Ny_cells * N_f
    xi = m0 * H + (np.arange(Nx_fine) + 0.5) * h
    yj = n0 * H + (np.arange(Ny_fine) + 0.5) * h
    XC, YC = np.meshgrid(xi, yj, indexing="xy")
    return XC.ravel(), YC.ravel(), Nx_fine, Ny_fine


class InteriorFactor:
    """Cached interior-submatrix factor for repeated Dirichlet solves.

    Solves A u = F with u[bdy] = bdy_values via
        A_ii u_i = F_i - A_ib u_b,   u_b = bdy_values
    using scipy.sparse.linalg.splu once per matrix.
    """

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


def _interior_solve(A: sp.csc_matrix, F: np.ndarray,
                    bdy: np.ndarray, bdy_values: np.ndarray) -> np.ndarray:
    """One-shot Dirichlet solve (no caching). Kept for tests and backward
    compatibility; for batch pipelines use `InteriorFactor`."""
    fac = InteriorFactor(A, bdy)
    return fac.solve(F, bdy, bdy_values)


# -----------------------------------------------------------------------------
# Public operators
# -----------------------------------------------------------------------------


def basefun(a_fun: Callable, m: int, n: int, N_c: int, N_f: int) -> sp.csc_matrix:
    """Q1 fine-scale stiffness on coarse cell (m,n). Matches Matlab basefun.m."""
    xc, yc = _cell_center_coords(m, n, N_c, N_f)
    a_vals = np.asarray(a_fun(xc, yc))
    return _assemble_q1_on_grid(a_vals, N_f, N_f)


class Workspace:
    """Pre-factorised stiffness cache for one ExpMsFEM run.

    Builds and stores:
      * cell_factor[(m, n)]: (B_sparse, bdy, InteriorFactor)  — per coarse cell
      * patch_factor[(m_edge, n_edge, t)]: (A_sparse, Nx, Ny, m0, n0, bdy, InteriorFactor)
        where (m_edge, n_edge) are the lower cell of the edge pair and t ∈ {1, 2}.
    """

    def __init__(self, a_fun: Callable, N_c: int, N_f: int):
        self.a_fun = a_fun
        self.N_c = N_c
        self.N_f = N_f
        self._cell = {}
        self._patch = {}
        self._edge_cache = {}  # populated by element_basis.prefactor_edges

    def _build_cell(self, m: int, n: int):
        B = basefun(self.a_fun, m, n, self.N_c, self.N_f)
        bdy = cell_boundary_nodes(self.N_f, self.N_f)
        return B, bdy, InteriorFactor(B, bdy)

    def _build_patch(self, m: int, n: int, t: int):
        A, Nx, Ny, m0, n0 = basefun1(self.a_fun, m, n, self.N_c, self.N_f, t)
        bdy = cell_boundary_nodes(Nx, Ny)
        return A, Nx, Ny, m0, n0, bdy, InteriorFactor(A, bdy)

    def cell_factor(self, m: int, n: int):
        key = (m, n)
        fac = self._cell.get(key)
        if fac is None:
            fac = self._build_cell(m, n)
            self._cell[key] = fac
        return fac

    def patch_factor(self, m: int, n: int, t: int):
        key = (m, n, t)
        fac = self._patch.get(key)
        if fac is None:
            fac = self._build_patch(m, n, t)
            self._patch[key] = fac
        return fac

    def prefactor_all(self, n_workers: int | None = None):
        """Factor every cell and every interior-edge patch up front in a
        thread pool. SuperLU factorisation and dense assembly release the GIL
        in C code, so threading scales to roughly min(n_workers, n_cores).
        """
        import concurrent.futures as cf

        N_c = self.N_c
        cell_keys = [(m, n) for n in range(N_c) for m in range(N_c)]
        # Horizontal patches: lower cell of pair has n_edge ∈ 0..N_c-2
        # Vertical patches:   lower cell of pair has m_edge ∈ 0..N_c-2
        patch_keys = [
            (m, n, 1) for n in range(N_c - 1) for m in range(N_c)
        ] + [
            (m, n, 2) for n in range(N_c) for m in range(N_c - 1)
        ]

        with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
            cell_jobs = {pool.submit(self._build_cell, m, n): (m, n)
                         for (m, n) in cell_keys}
            patch_jobs = {pool.submit(self._build_patch, m, n, t): (m, n, t)
                          for (m, n, t) in patch_keys}
            for fut in cf.as_completed(cell_jobs):
                self._cell[cell_jobs[fut]] = fut.result()
            for fut in cf.as_completed(patch_jobs):
                self._patch[patch_jobs[fut]] = fut.result()


def _cell_constant_rhs(N_c: int, N_f: int) -> np.ndarray:
    """Q1 load vector for RHS ≡ -1 on a single coarse cell."""
    H = 1.0 / N_c
    h = H / N_f
    n_nodes = (N_f + 1) ** 2
    G = local_to_global_nodes(N_f, N_f)
    F = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(F, G.ravel(), -(h**2) / 4)
    return F


def bubble(a_fun: Callable, m: int, n: int, N_c: int, N_f: int) -> np.ndarray:
    """Cell bubble: Q1 Dirichlet-0 solve with constant RHS = -1 on coarse cell
    (m,n). Returns a flat array of shape ((N_f+1)^2,). Matches Matlab bubble.m.
    """
    A = basefun(a_fun, m, n, N_c, N_f)
    bdy = cell_boundary_nodes(N_f, N_f)
    F = _cell_constant_rhs(N_c, N_f)
    return _interior_solve(A, F, bdy, np.zeros(bdy.size))


def bubble_cached(ws: "Workspace", m: int, n: int) -> np.ndarray:
    """Cell bubble using Workspace cache."""
    B, bdy, lu = ws.cell_factor(m, n)
    F = _cell_constant_rhs(ws.N_c, ws.N_f)
    return lu.solve(F, bdy, np.zeros(bdy.size))


def _patch_bounds_for_edge(m: int, n: int, N_c: int, t: int):
    """Return (m0, Nx_cells, n0, Ny_cells) for the oversampled patch used in
    Matlab basefun1 for the edge between cells (m, n) and (m, n+1) if t==1
    (horizontal), or (m+1, n) if t==2 (vertical). Inputs m, n are 0-indexed.

    Matches Matlab branches (note: Matlab m ranges 1..N_c-1 for interior edges;
    our 0-indexed m ranges 0..N_c-2).
    """
    if t == 1:
        # Horizontal edge (m, n) – (m, n+1). Oversample in x across columns
        # m-1, m, m+1; in y always 2 cells tall (n, n+1).
        if m == 0:
            m0, Nx = 0, 2
        elif m == N_c - 1:
            m0, Nx = N_c - 2, 2
        else:
            m0, Nx = m - 1, 3
        n0, Ny = n, 2
    else:
        # Vertical edge (m, n) – (m+1, n). Oversample in y across rows
        # n-1, n, n+1; in x always 2 cells wide (m, m+1).
        if n == 0:
            n0, Ny = 0, 2
        elif n == N_c - 1:
            n0, Ny = N_c - 2, 2
        else:
            n0, Ny = n - 1, 3
        m0, Nx = m, 2
    return m0, Nx, n0, Ny


def basefun1(a_fun: Callable, m: int, n: int, N_c: int, N_f: int, t: int):
    """Oversampled-patch Q1 stiffness around the edge of cells (m,n)–(m,n+1)
    (t=1, horizontal) or (m,n)–(m+1,n) (t=2, vertical). Matches basefun1.m.

    Returns (A, Nx_fine, Ny_fine, m0, n0) where Nx_fine = Nx_cells*N_f,
    Ny_fine = Ny_cells*N_f, and (m0, n0) is the lower-left coarse cell index.
    """
    m0, Nx_cells, n0, Ny_cells = _patch_bounds_for_edge(m, n, N_c, t)
    xc, yc, Nx_fine, Ny_fine = _patch_center_coords(
        m0, n0, Nx_cells, Ny_cells, N_c, N_f
    )
    a_vals = np.asarray(a_fun(xc, yc))
    A = _assemble_q1_on_grid(a_vals, Nx_fine, Ny_fine)
    return A, Nx_fine, Ny_fine, m0, n0


def _harmext_rhs_matrices(N_f: int, i: int):
    """Build identity Dirichlet-data matrices f1, f2 of shape (4*N_f, N_f-1)."""
    def identity_on(segment: slice):
        f = np.zeros((4 * N_f, N_f - 1), dtype=np.float64)
        idx = np.arange(4 * N_f)[segment]
        assert idx.size == N_f - 1, (idx.size, segment, 4 * N_f)
        f[idx, np.arange(N_f - 1)] = 1.0
        return f

    if i == 1:
        f1 = identity_on(slice(3 * N_f + 1, 4 * N_f))
        f2 = identity_on(slice(1, N_f))
    else:
        f1 = identity_on(slice(2 * N_f + 1, 3 * N_f))
        f2 = identity_on(slice(N_f + 1, 2 * N_f))
    return f1, f2


def harmext(a_fun: Callable, m: int, n: int, N_c: int, N_f: int, i: int):
    """Two-cell harmonic extension of identity-on-shared-edge data.

    For i==1 (horizontal shared edge between (m,n) and (m,n+1)):
      K1 = cell (m, n) stiffness,    Dirichlet data on its TOP edge interior
      K2 = cell (m, n+1) stiffness,  Dirichlet data on its BOTTOM edge interior

    For i==2 (vertical shared edge between (m,n) and (m+1,n)):
      K1 = cell (m, n) stiffness,    Dirichlet data on its RIGHT edge interior
      K2 = cell (m+1, n) stiffness,  Dirichlet data on its LEFT edge interior

    Returns
    -------
    L1, L2 : (N_f+1)^2 x (N_f-1)
        Each column is the harmonic lift on the respective cell of one basis
        vector of Dirichlet data (a hat function on the shared edge interior).
    N : (N_f-1) x (N_f-1)
        Sum of energies L1' K1 L1 + L2' K2 L2 (dense).
    """
    K1 = basefun(a_fun, m, n, N_c, N_f)
    if i == 1:
        K2 = basefun(a_fun, m, n + 1, N_c, N_f)
    else:
        K2 = basefun(a_fun, m + 1, n, N_c, N_f)

    n_nodes = (N_f + 1) ** 2
    b = cell_boundary_nodes(N_f, N_f)
    f1, f2 = _harmext_rhs_matrices(N_f, i)

    F_zero = np.zeros((n_nodes, N_f - 1), dtype=np.float64)
    L1 = _interior_solve(K1, F_zero, b, f1)
    L2 = _interior_solve(K2, F_zero, b, f2)
    N_mat = L1.T @ (K1 @ L1) + L2.T @ (K2 @ L2)
    return L1, L2, N_mat


def harmext_cached(ws: "Workspace", m: int, n: int, i: int):
    """Cached harmonic-extension using Workspace cell factors."""
    K1, b1, lu1 = ws.cell_factor(m, n)
    if i == 1:
        K2, b2, lu2 = ws.cell_factor(m, n + 1)
    else:
        K2, b2, lu2 = ws.cell_factor(m + 1, n)
    N_f = ws.N_f
    n_nodes = (N_f + 1) ** 2
    f1, f2 = _harmext_rhs_matrices(N_f, i)
    F_zero = np.zeros((n_nodes, N_f - 1), dtype=np.float64)
    L1 = lu1.solve(F_zero, b1, f1)
    L2 = lu2.solve(F_zero, b2, f2)
    N_mat = L1.T @ (K1 @ L1) + L2.T @ (K2 @ L2)
    return L1, L2, N_mat


def _restrict_active_mask(N_c: int, N_f: int, m: int, n: int, t: int,
                          N_x: int, N_y: int) -> np.ndarray:
    """Bool array of length 2*(N_x+N_y) indicating which patch-perimeter DOFs
    are usable Dirichlet DOFs. The last entry is always masked off (Matlab
    trims the last column of f). Patches abutting ∂Ω additionally mask off
    perimeter DOFs that sit on the domain boundary.

    Ported from restrict.m's nested if/elif ladder.
    """
    P_size = 2 * (N_x + N_y)
    mask = np.ones(P_size, dtype=bool)

    def zero_range_1idx(a, b):
        mask[a - 1 : b] = False

    M, Nn = m + 1, n + 1  # Matlab 1-indexed
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


def _edge_trace_indices(t: int, m: int, n: int, N_c: int, N_f: int,
                        Nx_fine: int, Ny_fine: int) -> np.ndarray:
    """Patch node indices for the N_f+1 nodes along the shared edge, in order."""
    if t == 1:
        if m == 0:
            col_start, col_end = 0, N_f + 1
        else:
            col_start, col_end = N_f, 2 * N_f + 1
        row_idx = N_f
        return row_idx * (Nx_fine + 1) + np.arange(col_start, col_end)
    else:
        if n == 0:
            row_start, row_end = 0, N_f + 1
        else:
            row_start, row_end = N_f, 2 * N_f + 1
        col_idx = N_f
        return np.arange(row_start, row_end) * (Nx_fine + 1) + col_idx


def _restrict_from_factors(N_c: int, N_f: int, m: int, n: int, t: int,
                           A, Nx_fine: int, Ny_fine: int,
                           b_patch: np.ndarray, lu_patch):
    """Shared body of restrict / restrict_cached. Given patch stiffness and
    its InteriorFactor (and patch boundary-node list), returns (R, P, bub)."""
    # Which perimeter DOFs to activate
    mask = _restrict_active_mask(N_c, N_f, m, n, t, Nx_fine, Ny_fine)
    active_idx = np.where(mask)[0]
    n_active = active_idx.size
    bdy_vals = np.zeros((b_patch.size, n_active), dtype=np.float64)
    bdy_vals[active_idx, np.arange(n_active)] = 1.0

    n_patch = (Nx_fine + 1) * (Ny_fine + 1)
    F_zero = np.zeros((n_patch, n_active), dtype=np.float64)
    harm = lu_patch.solve(F_zero, b_patch, bdy_vals)

    # Patch bubble
    H = 1.0 / N_c
    h = H / N_f
    G = local_to_global_nodes(Nx_fine, Ny_fine)
    F = np.zeros(n_patch, dtype=np.float64)
    np.add.at(F, G.ravel(), -(h**2) / 4)
    bub = lu_patch.solve(F, b_patch, np.zeros(b_patch.size))

    P = harm.T @ (A @ harm)

    edge_nodes = _edge_trace_indices(t, m, n, N_c, N_f, Nx_fine, Ny_fine)
    R_full = harm[edge_nodes, :]
    bub_edge = bub[edge_nodes]
    leng = N_f + 1
    lin_left = np.linspace(1, 0, leng)
    lin_right = np.linspace(0, 1, leng)
    R_full = R_full - np.outer(lin_left, R_full[0, :]) - np.outer(lin_right, R_full[-1, :])
    bub_edge = bub_edge - lin_left * bub_edge[0] - lin_right * bub_edge[-1]
    R = R_full[1 : leng - 1, :]
    bub_edge = bub_edge[1 : leng - 1]
    return R, P, bub_edge


def restrict_cached(ws: "Workspace", m: int, n: int, t: int):
    """Cached version of restrict using Workspace patch factors."""
    A, Nx, Ny, m0, n0, b_patch, lu = ws.patch_factor(m, n, t)
    return _restrict_from_factors(ws.N_c, ws.N_f, m, n, t, A, Nx, Ny, b_patch, lu)


def restrict(a_fun: Callable, m: int, n: int, N_c: int, N_f: int, t: int):
    """Oversampled-patch restriction for the edge between cells (m,n) and
    (m,n+1) if t==1 (horizontal) or (m+1,n) if t==2 (vertical).

    Returns
    -------
    R : (N_f-1, n_bdy_dof)
    P : (n_bdy_dof, n_bdy_dof)
    bub : (N_f-1,)

    `n_bdy_dof = 2*(Nx_fine + Ny_fine) - 1` generic-interior; fewer at ∂Ω.
    """
    A, Nx_fine, Ny_fine, m0, n0 = basefun1(a_fun, m, n, N_c, N_f, t)
    b_patch = cell_boundary_nodes(Nx_fine, Ny_fine)
    lu = InteriorFactor(A, b_patch)
    return _restrict_from_factors(N_c, N_f, m, n, t, A, Nx_fine, Ny_fine, b_patch, lu)
