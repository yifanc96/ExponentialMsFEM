"""Helmholtz local operators: complex Q1 patch stiffness, Dirichlet-set helper,
harmonic extensions, and oversampled-patch restriction.

All operators mirror the Matlab helmholtz/case1/Exp/*.m reference. Dirichlet
boundaries on the LOCAL patch now exclude any side that lies on ∂Ω: the Matlab
`bc(N_c_x, N_c_y, N_x, N_y, m, n)` helper strips bottom/top/left/right from the
default perimeter set according to whether the patch touches the domain boundary
on that side (which is gated by m, n, N_c_x, N_c_y). The impedance BC for the
domain-boundary sides stays baked into the Helmholtz stiffness K, so those DOFs
remain free in the system.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..mesh import cell_boundary_nodes, local_to_global_nodes
from . import coefficients as hc
from . import q1 as hq1


# -----------------------------------------------------------------------------
# Dirichlet-set helper (mirrors Matlab bc.m)
# -----------------------------------------------------------------------------


def dirichlet_indices(N_c_x: int, N_c_y: int, N_x: int, N_y: int,
                      m: int, n: int) -> np.ndarray:
    """Return the local-patch node indices that must be Dirichlet-fixed.

    Parameters (all 1-indexed to match Matlab bc.m exactly):
      N_c_x, N_c_y : number of coarse cells in x/y covered by the patch region
      N_x, N_y     : number of FINE elements in the patch in x/y
      m, n         : position of the local patch within the coarse grid (1-idx)

    `bc(N_c_x, N_c_y, N_x, N_y, m, n)` returns the perimeter node indices
    whose sides lie on ∂Ω — i.e. the sides we must pin to Dirichlet data.
    Sides that are interior to the coarse grid remain free (they get their
    natural Robin / impedance terms via the assembled stiffness).
    """
    Nxp1 = N_x + 1
    Nyp1 = N_y + 1
    # Full perimeter set (1-indexed), in Matlab's order:
    # [bottom | left-col interior | right-col interior | top-row interior]
    perim = np.concatenate([
        np.arange(1, Nxp1 + 1),
        np.arange(Nxp1 + 1, Nxp1 * Nyp1 + 1, Nxp1),
        np.arange(2 * Nxp1, Nxp1 * Nyp1 + 1, Nxp1),
        np.arange(N_x * N_y + N_y + 2, Nxp1 * Nyp1),
    ])
    a = set(perim.tolist())
    b = set(perim.tolist())

    if n > 1:
        # Bottom row is interior → strip it from b
        b -= set(np.arange(1, Nxp1 + 1).tolist())
    if n < N_c_y:
        # Top row interior → strip it from b
        b -= set(np.arange(N_x * N_y + N_y + 1, Nxp1 * Nyp1 + 1).tolist())
    if m > 1:
        # Left column interior → strip it from b
        b -= set(np.arange(1, Nxp1 * Nyp1 + 1, Nxp1).tolist())
    if m < N_c_x:
        # Right column interior → strip it from b
        b -= set(np.arange(Nxp1, Nxp1 * Nyp1 + 1, Nxp1).tolist())

    a -= b
    # Return 0-indexed, sorted
    return np.array(sorted(a), dtype=np.int64) - 1


# -----------------------------------------------------------------------------
# Q1 assembly on a fine grid of given size with Helmholtz element stiffness
# -----------------------------------------------------------------------------


def _cell_corners(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Build cell-corner array for a rectangular grid with fine nodes at xs
    (length Nx+1) and ys (length Ny+1), ordered j-major (row-by-row)."""
    Nx = xs.size - 1
    Ny = ys.size - 1
    rows = []
    for j in range(Ny):
        for i in range(Nx):
            rows.append([xs[i], xs[i + 1], ys[j], ys[j + 1]])
    return np.array(rows)


def _assemble_helm_stiffness(xs: np.ndarray, ys: np.ndarray,
                             k0: float) -> sp.csc_matrix:
    """Assemble the complex Helmholtz Q1 stiffness on a rectangular fine grid."""
    Nx = xs.size - 1
    Ny = ys.size - 1
    corners = _cell_corners(xs, ys)
    K_loc, _ = hq1.batched_element_matrices(corners, k0)
    G = local_to_global_nodes(Nx, Ny)
    n = (Nx + 1) * (Ny + 1)
    I = np.broadcast_to(G[:, :, None], (Nx * Ny, 4, 4)).reshape(-1)
    J = np.broadcast_to(G[:, None, :], (Nx * Ny, 4, 4)).reshape(-1)
    return sp.coo_matrix((K_loc.reshape(-1), (I, J)), shape=(n, n)).tocsc()


def _assemble_helm_load(xs: np.ndarray, ys: np.ndarray,
                        k0: float) -> np.ndarray:
    """Assemble the complex load vector for a rectangular fine grid."""
    Nx = xs.size - 1
    Ny = ys.size - 1
    corners = _cell_corners(xs, ys)
    f_loc = hq1.batched_element_loads(corners, k0)
    G = local_to_global_nodes(Nx, Ny)
    n = (Nx + 1) * (Ny + 1)
    F = np.zeros(n, dtype=np.complex128)
    np.add.at(F, G.ravel(), f_loc.reshape(-1))
    return F


# -----------------------------------------------------------------------------
# Factor cache for repeated Dirichlet solves on a complex matrix
# -----------------------------------------------------------------------------


class ComplexInteriorFactor:
    """Like the elliptic InteriorFactor but keeps a complex LU factor.

    Also handles the Helmholtz-style "use transpose" convention: solves
    `A.T u = F` by factoring A.T once (for direct-solve parity with Matlab's
    `A.' \\ F`). Pass `transpose=True` to get the transpose behaviour.
    """

    __slots__ = ("n", "interior", "A_ib", "lu", "transpose")

    def __init__(self, A: sp.csc_matrix, bdy: np.ndarray,
                 transpose: bool = False):
        n = A.shape[0]
        mask = np.ones(n, dtype=bool)
        mask[bdy] = False
        self.n = n
        self.interior = np.where(mask)[0]
        self.transpose = transpose
        A_csr = A.tocsr()
        A_ii = A_csr[self.interior][:, self.interior].tocsc()
        self.A_ib = A_csr[self.interior][:, bdy]
        if transpose:
            self.lu = spla.splu(A_ii.T.tocsc())
        else:
            self.lu = spla.splu(A_ii)

    def solve(self, F: np.ndarray, bdy: np.ndarray,
              bdy_values: np.ndarray) -> np.ndarray:
        rhs = F[self.interior] - self.A_ib @ bdy_values
        u_i = self.lu.solve(np.asarray(rhs, dtype=np.complex128))
        out_shape = (self.n,) if rhs.ndim == 1 else (self.n, rhs.shape[1])
        u = np.zeros(out_shape, dtype=np.complex128)
        u[self.interior] = u_i
        u[bdy] = bdy_values
        return u


# -----------------------------------------------------------------------------
# basefun / basefun1 / bubble
# -----------------------------------------------------------------------------


def _coarse_cell_grid(m: int, n: int, N_c: int, N_f: int):
    """Fine x/y coords for coarse cell (m, n), 0-indexed."""
    H = 1.0 / N_c
    xs = np.linspace(m * H, (m + 1) * H, N_f + 1)
    ys = np.linspace(n * H, (n + 1) * H, N_f + 1)
    return xs, ys


def _patch_bounds(m: int, n: int, N_c: int, t: int):
    """Mirror Matlab basefun1.m: oversampled patch around edge (m, n, t).
    Returns (m0, Nx_cells, n0, Ny_cells, N_c_x, N_c_y) where (N_c_x, N_c_y)
    are the "Matlab-equivalent" coarse-cell-count arguments fed to bc().

    The original Matlab passes `bc(N_c-2, N_c-1, N_x, N_y, m-1, n)` for t=1
    and `bc(N_c-1, N_c-2, ...)` for t=2 inside restrict.m. That's because
    `N_c_x`/`N_c_y` there are the number of coarse cells the patch spans
    MINUS ONE (for the x-direction), reflecting which sides are "interior to
    the patch" vs "on ∂Ω of the patch".

    We return the patch coord ranges plus (N_c_x_for_bc, N_c_y_for_bc) so
    the caller can pass to dirichlet_indices.
    """
    if t == 1:
        # Horizontal edge: x spans up to 3 cells, y spans 2 cells.
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


def basefun(m: int, n: int, N_c: int, N_f: int, k0: float) -> sp.csc_matrix:
    """Complex Q1 stiffness on coarse cell (m, n)."""
    xs, ys = _coarse_cell_grid(m, n, N_c, N_f)
    return _assemble_helm_stiffness(xs, ys, k0)


def basefun1(m: int, n: int, N_c: int, N_f: int, t: int, k0: float):
    """Oversampled-patch complex stiffness for edge (m, n, t)."""
    m0, Nx_cells, n0, Ny_cells = _patch_bounds(m, n, N_c, t)
    H = 1.0 / N_c
    xs = np.linspace(m0 * H, (m0 + Nx_cells) * H, Nx_cells * N_f + 1)
    ys = np.linspace(n0 * H, (n0 + Ny_cells) * H, Ny_cells * N_f + 1)
    A = _assemble_helm_stiffness(xs, ys, k0)
    return A, xs.size - 1, ys.size - 1, m0, n0


def bubble(m: int, n: int, N_c: int, N_f: int, k0: float) -> np.ndarray:
    """Cell bubble: Helmholtz Dirichlet solve on one coarse cell with the
    true ffun/gfun-based load. Matlab bubble.m pins the Dirichlet set via
    bc(N_c, N_c, N_f, N_f, m, n) so boundary-cell edges on ∂Ω stay free.
    """
    xs, ys = _coarse_cell_grid(m, n, N_c, N_f)
    A = _assemble_helm_stiffness(xs, ys, k0)
    F = _assemble_helm_load(xs, ys, k0)
    bdy = dirichlet_indices(N_c, N_c, N_f, N_f, m + 1, n + 1)
    bdy_vals = np.zeros(bdy.size, dtype=np.complex128)

    # Matlab: u = A.' \ F with Dirichlet elimination already applied.
    # Use transpose convention:
    fac = ComplexInteriorFactor(A, bdy, transpose=True)
    return fac.solve(F, bdy, bdy_vals)


# -----------------------------------------------------------------------------
# Workspace cache and pre-factor helpers
# -----------------------------------------------------------------------------


class HelmWorkspace:
    """Caches complex LU factors for Helmholtz cells and patches."""

    def __init__(self, N_c: int, N_f: int, k0: float):
        self.N_c = N_c
        self.N_f = N_f
        self.k0 = k0
        self._cell = {}       # (m, n) -> (A, bdy, factor, factor_T)
        self._patch = {}      # (m, n, t) -> (A, Nx, Ny, m0, n0, bdy, factor, factor_T)
        self._edge_cache = {} # (t, m_edge, n_edge) -> (L1_RV, L2_RV, L1_bub, L2_bub)

    def _build_cell(self, m: int, n: int):
        A = basefun(m, n, self.N_c, self.N_f, self.k0)
        bdy = dirichlet_indices(self.N_c, self.N_c, self.N_f, self.N_f,
                                m + 1, n + 1)
        fac = ComplexInteriorFactor(A, bdy, transpose=False)
        fac_T = ComplexInteriorFactor(A, bdy, transpose=True)
        return (A, bdy, fac, fac_T)

    def _build_patch(self, m: int, n: int, t: int):
        A, Nx, Ny, m0, n0 = basefun1(m, n, self.N_c, self.N_f, t, self.k0)
        # bc for the patch: use Matlab's restrict.m convention
        #   t==1 → bc(N_c-2, N_c-1, N_x, N_y, m-1, n)
        #   t==2 → bc(N_c-1, N_c-2, N_x, N_y, m, n-1)
        if t == 1:
            bdy = dirichlet_indices(self.N_c - 2, self.N_c - 1, Nx, Ny,
                                    (m + 1) - 1, n + 1)
        else:
            bdy = dirichlet_indices(self.N_c - 1, self.N_c - 2, Nx, Ny,
                                    m + 1, (n + 1) - 1)
        fac = ComplexInteriorFactor(A, bdy, transpose=False)
        fac_T = ComplexInteriorFactor(A, bdy, transpose=True)
        return (A, Nx, Ny, m0, n0, bdy, fac, fac_T)

    def cell_factor(self, m: int, n: int):
        key = (m, n)
        f = self._cell.get(key)
        if f is None:
            f = self._build_cell(m, n)
            self._cell[key] = f
        return f

    def patch_factor(self, m: int, n: int, t: int):
        key = (m, n, t)
        f = self._patch.get(key)
        if f is None:
            f = self._build_patch(m, n, t)
            self._patch[key] = f
        return f

    def prefactor_all(self, n_workers: int | None = None):
        import concurrent.futures as cf

        N_c = self.N_c
        cell_keys = [(m, n) for n in range(N_c) for m in range(N_c)]
        patch_keys = (
            [(m, n, 1) for n in range(N_c - 1) for m in range(N_c)]
            + [(m, n, 2) for n in range(N_c) for m in range(N_c - 1)]
        )
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
# harmext (complex harmonic extension between two adjacent cells)
# -----------------------------------------------------------------------------


def _shared_edge_identity_rhs(N_f: int, t: int, bdy_local: np.ndarray,
                              side_flag: str) -> np.ndarray:
    """Build Dirichlet data matrix of shape (|bdy_local|, N_f-1) that puts
    the identity on the *shared edge* interior nodes (the N_f-1 interior
    nodes of the edge that joins the cell to its neighbour).

    side_flag ∈ {"top", "bottom", "right", "left"} picks which edge of the
    local cell is the shared one. Only the entries of bdy_local that lie on
    that edge will be set; others stay zero.
    """
    N_f_nodes_per_side = N_f + 1
    # Compute the global (full-perimeter) indices of the shared-edge interior
    # nodes.
    if side_flag == "top":
        # Nodes (i, N_f) for i = 1..N_f-1 → j-major indices:
        #   j = N_f, i = 1..N_f-1 → idx = N_f * (N_f+1) + i
        target = N_f * (N_f + 1) + np.arange(1, N_f)
    elif side_flag == "bottom":
        # Nodes (i, 0) for i = 1..N_f-1 → idx = i
        target = np.arange(1, N_f)
    elif side_flag == "right":
        # Nodes (N_f, j) for j = 1..N_f-1 → idx = j * (N_f+1) + N_f
        target = np.arange(1, N_f) * (N_f + 1) + N_f
    elif side_flag == "left":
        # Nodes (0, j) for j = 1..N_f-1 → idx = j * (N_f+1)
        target = np.arange(1, N_f) * (N_f + 1)
    else:
        raise ValueError(side_flag)

    # We need to place identity in rows of `bdy_local` that correspond to
    # each target index. Build a mapping.
    pos = {int(b): k for k, b in enumerate(bdy_local)}
    data = np.zeros((bdy_local.size, N_f - 1), dtype=np.complex128)
    col = 0
    for i, idx in enumerate(target):
        row = pos[int(idx)]
        data[row, col] = 1.0
        col += 1
    return data


def harmext(ws: HelmWorkspace, m: int, n: int, t: int):
    """Return (L1, L2, N) for the shared edge between (m, n) and its
    t-neighbour. Mirrors Matlab harmext.m for Helmholtz.

    L1 lives on cell (m, n); L2 lives on the neighbour (m, n+1) if t=1 or
    (m+1, n) if t=2. Each column corresponds to a hat function on the shared
    edge interior (N_f-1 of them).
    """
    N_f = ws.N_f
    A1, b1, _, lu1_T = ws.cell_factor(m, n)

    if t == 1:
        A2, b2, _, lu2_T = ws.cell_factor(m, n + 1)
        f1 = _shared_edge_identity_rhs(N_f, t, b1, "top")
        f2 = _shared_edge_identity_rhs(N_f, t, b2, "bottom")
    else:
        A2, b2, _, lu2_T = ws.cell_factor(m + 1, n)
        f1 = _shared_edge_identity_rhs(N_f, t, b1, "right")
        f2 = _shared_edge_identity_rhs(N_f, t, b2, "left")

    F_zero = np.zeros(((N_f + 1) ** 2, N_f - 1), dtype=np.complex128)

    # Matlab solves K1 \ F1 (NON-transpose) in harmext.m for harmonic extension
    # (the "harmonic" relative to the Helmholtz K acting as a primary equation,
    # not its adjoint). Use the non-transpose factor.
    _, _, lu1, _ = ws.cell_factor(m, n)
    if t == 1:
        _, _, lu2, _ = ws.cell_factor(m, n + 1)
    else:
        _, _, lu2, _ = ws.cell_factor(m + 1, n)

    L1 = lu1.solve(F_zero, b1, f1)
    L2 = lu2.solve(F_zero, b2, f2)

    # N = L1' K1 L1 + L2' K2 L2   (Matlab uses ' which is conjugate transpose)
    N_mat = L1.conj().T @ (A1 @ L1) + L2.conj().T @ (A2 @ L2)
    return L1, L2, N_mat


# -----------------------------------------------------------------------------
# restrict (oversampled patch → edge trace + energy Gram + edge bubble)
# -----------------------------------------------------------------------------


def restrict(ws: HelmWorkspace, m: int, n: int, t: int):
    """Returns (R, P, bub) for the edge (m, n, t).

    R : (N_f-1, n_active)  — patch harmonic extension restricted to edge
                              (minus linear-endpoint part, minus two endpoints)
    P : (n_active, n_active) — complex energy Gram (harm.T @ A @ conj(harm))
    bub : (N_f-1,)         — patch-bubble edge trace, same treatment as R
    """
    A, Nx, Ny, m0, n0, b_patch, lu, lu_T = ws.patch_factor(m, n, t)
    N_c = ws.N_c
    N_f = ws.N_f
    # Mirror Matlab restrict.m boundary-set mask
    M = m + 1  # 1-indexed
    Nn = n + 1
    fully_interior = (
        (t == 1 and 2 < M < N_c - 1 and 1 < Nn < N_c - 1)
        or (t == 2 and 2 < Nn < N_c - 1 and 1 < M < N_c - 1)
    )
    if fully_interior:
        n_perimeter_total = 2 * (Nx + Ny)
        n_active = n_perimeter_total - 1
        f = np.zeros((b_patch.size, n_active), dtype=np.complex128)
        # Matlab: f(1:2*(Nx+Ny)-1, :) = eye(2*(Nx+Ny)-1)
        # But Matlab's b indexes a DIFFERENT set than our full perimeter when the
        # patch is interior; in that case `b` from bc(...) equals the full
        # (Nx+Ny) perimeter (since no side is on ∂Ω). We place identity on the
        # first n_active entries of b_patch (=> skipping the last).
        f[:n_active, np.arange(n_active)] = np.eye(n_active)
    else:
        # Patch touches ∂Ω: b_patch already strips the Robin sides; Matlab uses
        # speye(length(b)) — so every b_patch DOF is an active Dirichlet source.
        n_active = b_patch.size
        f = np.eye(n_active, dtype=np.complex128)

    # Harmonic extension solve
    F_zero = np.zeros(((Nx + 1) * (Ny + 1), n_active), dtype=np.complex128)
    harm = lu.solve(F_zero, b_patch, f)

    # Patch bubble from the Helmholtz RHS (ffun + gfun sides)
    # Build xs, ys for the patch and assemble load.
    H = 1.0 / N_c
    xs = np.linspace(m0 * H, (m0 + (Nx // N_f)) * H, Nx + 1)
    ys = np.linspace(n0 * H, (n0 + (Ny // N_f)) * H, Ny + 1)
    G = _assemble_helm_load(xs, ys, ws.k0)
    bub_full = lu.solve(G, b_patch, np.zeros(b_patch.size, dtype=np.complex128))

    # P = harm.' * A * conj(harm)   (Matlab sesquilinear form)
    P = harm.T @ (A @ harm.conj())
    P = 0.5 * (P + P.T)  # symmetrise for numerical robustness

    # Identify edge nodes in the patch
    if t == 1:
        if m == 0:
            col_start, col_end = 0, N_f + 1
        else:
            col_start, col_end = N_f, 2 * N_f + 1
        row_idx = N_f
        edge_nodes = row_idx * (Nx + 1) + np.arange(col_start, col_end)
    else:
        if n == 0:
            row_start, row_end = 0, N_f + 1
        else:
            row_start, row_end = N_f, 2 * N_f + 1
        col_idx = N_f
        edge_nodes = np.arange(row_start, row_end) * (Nx + 1) + col_idx

    leng = N_f + 1
    R_full = harm[edge_nodes, :]
    bub_edge = bub_full[edge_nodes]
    lin_left = np.linspace(1, 0, leng)
    lin_right = np.linspace(0, 1, leng)
    R_full = (R_full
              - np.outer(lin_left, R_full[0, :])
              - np.outer(lin_right, R_full[-1, :]))
    bub_edge = (bub_edge
                - lin_left * bub_edge[0]
                - lin_right * bub_edge[-1])
    R = R_full[1 : leng - 1, :]
    bub_edge = bub_edge[1 : leng - 1]

    # Matlab fallback: if the bubble trace is identically zero (as happens when
    # f=0 and the whole patch is in the interior), use the reference parabola
    # bub[i] = i(leng-1-i)/(leng-1)² as a placeholder on the edge — this keeps
    # the extra basis column from being a zero vector (which would make the
    # element stiffness singular).
    if np.linalg.norm(bub_edge) == 0:
        leng_m = leng - 1
        idx = np.arange(1, leng - 1)
        bub_edge = (idx * (leng_m - idx) / (leng_m ** 2)).astype(np.complex128)

    return R, P, bub_edge
