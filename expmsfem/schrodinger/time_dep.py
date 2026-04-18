"""Time-dependent Schrödinger via backward Euler + ExpMsFEM (Dirichlet-0
or periodic BC — select with `SemiclassicalParam.boundary`).

Semi-classical Schrödinger on `Ω = (0, 1)²`:

    i ε ∂_t ψ = -½ ε² Δ ψ + V(x) ψ,   ψ(0, x) = ψ₀(x),
    either  ψ|_∂Ω = 0         (boundary="dirichlet")
    or      ψ periodic in x, y (boundary="periodic")

Backward-Euler in time gives a linear complex elliptic problem per step:

    ( -½ ε² Δ + V(x) - i ε / Δt ) ψ_{n+1} = -i ε / Δt · ψ_n

with bilinear form

    b(ψ, φ) = ½ ε² (∇ψ, ∇φ̄) + (V ψ, φ̄) - i ε / Δt (ψ, φ̄).

The operator matrix is **complex-symmetric** (`B = Bᵀ`) for real `V`, so we
use the Matlab sesquilinear convention inherited from `expmsfem.helmholtz`:

    B_coarse = value.T · B_fine · conj(value)
    f_coarse = value.T · F_fine
    solve B_coarse · c = f_coarse

The offline phase (per-cell and per-patch LU factors of `B_cell`, edge
eigenbasis, edge-bubble) is paid **once** when `Δt` is held fixed — every
time step is then a cheap online coarse solve plus per-cell bubble
correction from the ψ_n-driven RHS. This is the harmonic-bubble
decomposition from the ExpMsFEM paper, instantiated with the shifted
complex operator.

See arXiv 2212.00823 (elliptic) and the Schrödinger note for the
theorem: exponential convergence in the number of edge modes `N_e` under
`H ≤ C₀ ε` and `H ≤ C₁ √(ε Δt)`.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..assembly import cell_global_indices, n_total, domain_boundary_dofs
from ..mesh import (cell_boundary_nodes, domain_boundary_nodes,
                    local_to_global_nodes)
from ..q1 import C_MASS, C_STIFF


# -----------------------------------------------------------------------------
# Physical parameters
# -----------------------------------------------------------------------------


@dataclass
class SemiclassicalParam:
    eps: float
    V_fun: Callable
    dt: float
    boundary: str = "dirichlet"   # "dirichlet" or "periodic"

    def __post_init__(self):
        if self.boundary not in ("dirichlet", "periodic"):
            raise ValueError(f"boundary must be 'dirichlet' or 'periodic', got {self.boundary!r}")

    @property
    def im_shift(self) -> complex:
        """The complex coefficient `-i ε / Δt` of the mass-matrix shift."""
        return -1j * self.eps / self.dt

    def V_at(self, x, y):
        """Evaluate V, wrapping coordinates to `[0, 1]²` when `boundary == "periodic"`.
        This lets the user supply an arbitrary V; the wrapping makes sure cells in
        oversampled patches that slightly exceed the domain still sample a
        sensible value (naturally consistent if V is already period-1)."""
        if self.boundary == "periodic":
            return self.V_fun(np.asarray(x) % 1.0, np.asarray(y) % 1.0)
        return self.V_fun(x, y)


# -----------------------------------------------------------------------------
# Q1 assembly — complex shifted operator B and real mass M on any grid
# -----------------------------------------------------------------------------


def _assemble_B_and_M(param: SemiclassicalParam,
                      xs: np.ndarray, ys: np.ndarray):
    """Return complex `B = (ε²/2)·K_stiff + V·M - (iε/Δt)·M` and real mass
    matrix `M` on the fine grid defined by node coordinates `xs, ys`."""
    Nx = xs.size - 1
    Ny = ys.size - 1
    h = xs[1] - xs[0]
    h2 = h * h
    # Fine-cell midpoints, j-major ordering
    xmid = 0.5 * (xs[:-1] + xs[1:])
    ymid = 0.5 * (ys[:-1] + ys[1:])
    XC, YC = np.meshgrid(xmid, ymid, indexing="xy")
    V_c = np.asarray(param.V_at(XC, YC), dtype=np.float64).ravel()

    eps2 = param.eps ** 2
    shift = param.im_shift            # = -i ε / Δt

    # Per-cell 4x4: (eps²/2)·C_STIFF + (V + shift)·h²·C_MASS
    coef = (V_c + shift)[:, None, None]   # complex
    K_loc = (0.5 * eps2) * C_STIFF[None, :, :] + coef * h2 * C_MASS[None, :, :]
    M_loc = np.broadcast_to(h2 * C_MASS, (Nx * Ny, 4, 4))

    G = local_to_global_nodes(Nx, Ny)
    n = (Nx + 1) * (Ny + 1)
    I = np.broadcast_to(G[:, :, None], (Nx * Ny, 4, 4)).reshape(-1)
    J = np.broadcast_to(G[:, None, :], (Nx * Ny, 4, 4)).reshape(-1)

    B = sp.coo_matrix((K_loc.reshape(-1), (I, J)),
                      shape=(n, n), dtype=np.complex128).tocsc()
    M = sp.coo_matrix((M_loc.reshape(-1), (I, J)),
                      shape=(n, n)).tocsc()
    return B, M


# -----------------------------------------------------------------------------
# Fine-scale reference — backward-Euler on Q1 grid
# -----------------------------------------------------------------------------


def _fine_periodic_remap(N_f: int) -> np.ndarray:
    """Return `remap` of length `(N_f+1)²` such that `remap[k_full]` is the
    periodic DOF index in `[0, N_f²)` for the original fine node `k_full`.
    Nodes at `i = N_f` are identified with `i = 0`, and similarly in `j`."""
    n = (N_f + 1) ** 2
    remap = np.empty(n, dtype=np.int64)
    for j in range(N_f + 1):
        for i in range(N_f + 1):
            remap[j * (N_f + 1) + i] = (j % N_f) * N_f + (i % N_f)
    return remap


def _reduce_to_periodic(A_full: sp.spmatrix, N_f: int):
    """Fold opposite-face fine DOFs into one via the gather matrix `P` so
    `A_per = P · A_full · P^T`  has shape `(N_f², N_f²)`."""
    remap = _fine_periodic_remap(N_f)
    n_full = (N_f + 1) ** 2
    n_per = N_f * N_f
    P = sp.csr_matrix((np.ones(n_full), (remap, np.arange(n_full))),
                      shape=(n_per, n_full))
    return (P @ A_full @ P.T).tocsc(), remap


def solve_fine_backward_euler(param: SemiclassicalParam, psi0: np.ndarray,
                              N_f: int, n_steps: int, save_stride: int = 1):
    """Backward-Euler fine-reference propagator. Returns

        ts        : (n_saved,) time values (including t=0)
        frames    : ((N_f+1)², n_saved) complex fine-scale snapshots of ψ
                    (re-expanded to the full `(N_f+1)²` grid in the periodic
                    case — opposite-face nodes share the same value).
        B, M      : the assembled sparse matrices (reduced-DOF for periodic,
                    full-DOF for Dirichlet).

    The operator is LU-factored once via `scipy.sparse.linalg.splu`; every
    step is a back-substitution.
    """
    xs = np.linspace(0, 1, N_f + 1)
    ys = xs.copy()
    B_full, M_full = _assemble_B_and_M(param, xs, ys)
    shift = param.im_shift

    if param.boundary == "periodic":
        B, remap = _reduce_to_periodic(B_full, N_f)
        M, _ = _reduce_to_periodic(M_full, N_f)
        lu = spla.splu(B)

        # Initial condition: pick canonical representatives (the N_f² nodes
        # with i, j in [0, N_f)). Assumes ψ₀ is already periodic (or effectively
        # zero near the boundary, like a centred wavepacket).
        psi_full0 = np.asarray(psi0, dtype=np.complex128).ravel()
        canonical = np.array([j * (N_f + 1) + i
                              for j in range(N_f) for i in range(N_f)],
                             dtype=np.int64)
        psi_per = psi_full0[canonical].copy()

        ts = [0.0]
        frames_full = [psi_per[remap].copy()]
        for step in range(1, n_steps + 1):
            rhs = shift * (M @ psi_per)
            psi_per = lu.solve(rhs)
            if step % save_stride == 0:
                ts.append(step * param.dt)
                frames_full.append(psi_per[remap].copy())
        return np.array(ts), np.array(frames_full).T, B, M

    # --- Dirichlet branch (unchanged) ---
    B = B_full
    M = M_full
    bdy = domain_boundary_nodes(N_f, N_f)
    mask = np.ones(B.shape[0], dtype=bool)
    mask[bdy] = False
    interior = np.where(mask)[0]
    B_int = B[interior][:, interior].tocsc()
    lu = spla.splu(B_int)

    psi = np.asarray(psi0, dtype=np.complex128).copy()
    psi[bdy] = 0.0

    ts = [0.0]
    frames = [psi.copy()]
    for step in range(1, n_steps + 1):
        rhs = shift * (M @ psi)
        rhs[bdy] = 0.0
        psi_int = lu.solve(rhs[interior])
        psi = np.zeros(B.shape[0], dtype=np.complex128)
        psi[interior] = psi_int
        if step % save_stride == 0:
            ts.append(step * param.dt)
            frames.append(psi.copy())
    return np.array(ts), np.array(frames).T, B, M


# -----------------------------------------------------------------------------
# Complex Dirichlet-solve factor (cached)
# -----------------------------------------------------------------------------


class ComplexInteriorFactor:
    """LU factor of `A[interior, interior]` for a complex sparse matrix,
    supporting fast Dirichlet solves with arbitrary boundary data."""

    __slots__ = ("n", "interior", "bdy", "A_ib", "lu")

    def __init__(self, A: sp.csc_matrix, bdy: np.ndarray):
        n = A.shape[0]
        mask = np.ones(n, dtype=bool)
        mask[bdy] = False
        self.n = n
        self.interior = np.where(mask)[0]
        self.bdy = np.asarray(bdy, dtype=np.int64)
        A_csr = A.tocsr()
        A_ii = A_csr[self.interior][:, self.interior].tocsc()
        self.A_ib = A_csr[self.interior][:, self.bdy]
        self.lu = spla.splu(A_ii)

    def solve(self, F: np.ndarray, bdy_values: np.ndarray) -> np.ndarray:
        rhs = F[self.interior] - self.A_ib @ bdy_values
        u_i = self.lu.solve(np.asarray(rhs, dtype=np.complex128))
        out_shape = (self.n,) if rhs.ndim == 1 else (self.n, rhs.shape[1])
        u = np.zeros(out_shape, dtype=np.complex128)
        u[self.interior] = u_i
        u[self.bdy] = bdy_values
        return u


# -----------------------------------------------------------------------------
# Per-cell and per-patch complex operators + Workspace
# -----------------------------------------------------------------------------


def cell_B_and_M(param: SemiclassicalParam, m: int, n: int, N_c: int, N_f: int):
    """Complex `B_cell` and real `M_cell` for coarse cell (m, n)."""
    H = 1.0 / N_c
    xs = np.linspace(m * H, (m + 1) * H, N_f + 1)
    ys = np.linspace(n * H, (n + 1) * H, N_f + 1)
    return _assemble_B_and_M(param, xs, ys)


def _patch_bounds(m: int, n: int, N_c: int, t: int):
    """3-wide × 2-wide (or shrunk at ∂Ω) oversampled-patch geometry,
    identical to the elliptic code."""
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


def _patch_bounds_periodic(m: int, n: int, N_c: int, t: int):
    """3-wide × 2-wide oversampled-patch geometry for periodic BC. All
    edges are interior (no domain boundary), so patches always extend a
    full cell on each side of the shared edge — m0 / n0 may be negative
    (meaning the patch wraps), which is fine as long as V is evaluated
    with modular coordinates."""
    if t == 1:
        m0 = m - 1            # may be -1 at m=0 → patch spans x ∈ [-H, 2H]
        Nx_cells = 3
        n0 = n
        Ny_cells = 2
    else:
        m0 = m
        Nx_cells = 2
        n0 = n - 1
        Ny_cells = 3
    return m0, Nx_cells, n0, Ny_cells


def patch_B_and_M(param: SemiclassicalParam, m: int, n: int, N_c: int,
                  N_f: int, t: int):
    """Complex `B_patch` and real `M_patch` for the oversampled patch around
    the edge at (m, n, t). Geometry depends on `param.boundary`."""
    if param.boundary == "periodic":
        m0, Nx_cells, n0, Ny_cells = _patch_bounds_periodic(m, n, N_c, t)
    else:
        m0, Nx_cells, n0, Ny_cells = _patch_bounds(m, n, N_c, t)
    H = 1.0 / N_c
    xs = np.linspace(m0 * H, (m0 + Nx_cells) * H, Nx_cells * N_f + 1)
    ys = np.linspace(n0 * H, (n0 + Ny_cells) * H, Ny_cells * N_f + 1)
    # V is evaluated via `param.V_at`, which wraps coords modulo 1 when periodic.
    B, M = _assemble_B_and_M(param, xs, ys)
    return B, M, Nx_cells * N_f, Ny_cells * N_f, m0, n0


class Workspace:
    """Caches complex LU factors for cells and oversampled patches.

    `prefactor_all` assembles and factors every cell and every interior
    edge patch in a thread pool; afterwards `cell_factor` /
    `patch_factor` are O(1) dict lookups.
    """

    def __init__(self, param: SemiclassicalParam, N_c: int, N_f: int):
        self.param = param
        self.N_c = N_c
        self.N_f = N_f
        self._cell = {}    # (m, n) → (B, M, bdy, ComplexInteriorFactor)
        self._patch = {}   # (m, n, t) → (B, M, Nx, Ny, m0, n0, bdy, factor)
        self._edge_cache = {}

    def _build_cell(self, m, n):
        B, M = cell_B_and_M(self.param, m, n, self.N_c, self.N_f)
        bdy = cell_boundary_nodes(self.N_f, self.N_f)
        return (B, M, bdy, ComplexInteriorFactor(B, bdy))

    def _build_patch(self, m, n, t):
        B, M, Nx, Ny, m0, n0 = patch_B_and_M(self.param, m, n,
                                             self.N_c, self.N_f, t)
        bdy = cell_boundary_nodes(Nx, Ny)
        return (B, M, Nx, Ny, m0, n0, bdy, ComplexInteriorFactor(B, bdy))

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

    def prefactor_all(self, n_workers=None):
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
# Harmonic extension and oversampled restriction (mirrors elliptic geometry,
# but with the complex shifted operator instead of the real Laplacian)
# -----------------------------------------------------------------------------


def _harmext_rhs(N_f: int, i: int):
    def identity_on(segment):
        f = np.zeros((4 * N_f, N_f - 1), dtype=np.complex128)
        idx = np.arange(4 * N_f)[segment]
        f[idx, np.arange(N_f - 1)] = 1.0
        return f
    if i == 1:
        return (identity_on(slice(3 * N_f + 1, 4 * N_f)),
                identity_on(slice(1, N_f)))
    return (identity_on(slice(2 * N_f + 1, 3 * N_f)),
            identity_on(slice(N_f + 1, 2 * N_f)))


def harmext(ws: Workspace, m, n, i):
    """Two-cell B-harmonic extension of identity shared-edge-interior hat
    data. Returns (L1, L2, N_mat) analogous to the elliptic code, where
    N_mat = L1' B1 L1 + L2' B2 L2 is the 2-cell complex energy Gram matrix
    (uses the Hermitian inner product L^H · B · L per Matlab Helmholtz
    convention). For `boundary == "periodic"` the neighbouring cell index
    wraps modulo `N_c`."""
    B1, _, b1, lu1 = ws.cell_factor(m, n)
    if ws.param.boundary == "periodic":
        other = (m, (n + 1) % ws.N_c) if i == 1 else ((m + 1) % ws.N_c, n)
    else:
        other = (m, n + 1) if i == 1 else (m + 1, n)
    B2, _, b2, lu2 = ws.cell_factor(*other)
    f1, f2 = _harmext_rhs(ws.N_f, i)
    n_nodes = (ws.N_f + 1) ** 2
    F_zero = np.zeros((n_nodes, ws.N_f - 1), dtype=np.complex128)
    L1 = lu1.solve(F_zero, f1)
    L2 = lu2.solve(F_zero, f2)
    N_mat = L1.conj().T @ (B1 @ L1) + L2.conj().T @ (B2 @ L2)
    return L1, L2, N_mat


def _restrict_active_mask(N_c, N_f, m, n, t, N_x, N_y):
    """Same patch-boundary ladder as the elliptic code (8 branches for
    domain-boundary-touching patches); the last perimeter DOF is always
    dropped."""
    P_size = 2 * (N_x + N_y)
    mask = np.ones(P_size, dtype=bool)

    def z(a, b):
        mask[a - 1:b] = False

    M, Nn = m + 1, n + 1
    Nc = N_c
    interior_m = 1 < M < Nc
    interior_n = 1 < Nn < Nc

    if t == 1:
        if Nn == 1 and (2 < M < Nc - 1):
            z(1, N_x + 1)
        elif Nn == Nc - 1 and (2 < M < Nc - 1):
            z(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
        elif (M < 3) and interior_n:
            z(N_x + 2, N_x + N_y + 1)
        elif (M > Nc - 2) and interior_n:
            z(N_x + N_y + 2, N_x + 2 * N_y + 1)
        elif Nn == 1 and M < 3:
            z(1, N_x + 1); z(N_x + 2, N_x + N_y + 1)
        elif Nn == 1 and M > Nc - 2:
            z(1, N_x + 1); z(N_x + N_y + 2, N_x + 2 * N_y + 1)
        elif Nn == Nc - 1 and M < 3:
            z(N_x + 2, N_x + N_y + 1); z(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
        elif Nn == Nc - 1 and M > Nc - 2:
            z(N_x + N_y + 2, N_x + 2 * N_y + 1)
            z(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
    else:
        if M == 1 and (2 < Nn < Nc - 1):
            z(N_x + 2, N_x + N_y + 1)
        elif M == Nc - 1 and (2 < Nn < Nc - 1):
            z(N_x + N_y + 2, N_x + 2 * N_y + 1)
        elif (Nn < 3) and interior_m:
            z(1, N_x + 1)
        elif (Nn > Nc - 2) and interior_m:
            z(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
        elif M == 1 and Nn < 3:
            z(N_x + 2, N_x + N_y + 1); z(1, N_x + 1)
        elif M == Nc - 1 and Nn < 3:
            z(N_x + N_y + 2, N_x + 2 * N_y + 1); z(1, N_x + 1)
        elif M == 1 and Nn > Nc - 2:
            z(N_x + 2, N_x + N_y + 1); z(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
        elif M == Nc - 1 and Nn > Nc - 2:
            z(N_x + N_y + 2, N_x + 2 * N_y + 1)
            z(N_x + 2 * N_y + 2, 2 * N_x + 2 * N_y)
    mask[-1] = False
    return mask


def _edge_trace_indices(t, m, n, N_c, N_f, Nx, Ny):
    if t == 1:
        if m == 0:
            col_start, col_end = 0, N_f + 1
        else:
            col_start, col_end = N_f, 2 * N_f + 1
        row_idx = N_f
        return row_idx * (Nx + 1) + np.arange(col_start, col_end)
    if n == 0:
        row_start, row_end = 0, N_f + 1
    else:
        row_start, row_end = N_f, 2 * N_f + 1
    col_idx = N_f
    return np.arange(row_start, row_end) * (Nx + 1) + col_idx


def restrict(ws: Workspace, m, n, t):
    """Oversampled-patch B-harmonic lifts restricted to the shared edge.

    Returns `(R, P, bub_edge)`:
      R       : (N_f-1, n_active) edge traces of the active patch-harmonic lifts
                with linear-endpoint part subtracted and two endpoint rows dropped
      P       : (n_active, n_active) complex energy Gram matrix harm' B harm
      bub_edge: (N_f-1,) edge trace of the patch bubble under constant RHS `f=1`
                (a *fixed* prototype used as an edge-bubble basis column —
                independent of the time-dependent ψ_n)
    """
    B_patch, _, Nx, Ny, m0, n0, b_patch, lu = ws.patch_factor(m, n, t)
    if ws.param.boundary == "periodic":
        # Every patch is fully interior to the torus — no domain-boundary
        # sides to mask off; just drop the last perimeter DOF.
        P_size = 2 * (Nx + Ny)
        mask = np.ones(P_size, dtype=bool)
        mask[-1] = False
    else:
        mask = _restrict_active_mask(ws.N_c, ws.N_f, m, n, t, Nx, Ny)
    active = np.where(mask)[0]
    n_active = active.size
    bdy_vals = np.zeros((b_patch.size, n_active), dtype=np.complex128)
    bdy_vals[active, np.arange(n_active)] = 1.0

    n_patch = (Nx + 1) * (Ny + 1)
    F_zero = np.zeros((n_patch, n_active), dtype=np.complex128)
    harm = lu.solve(F_zero, bdy_vals)

    # Fixed prototype edge bubble: constant f = 1 (mass-lumped h²/4 per corner)
    H = 1.0 / ws.N_c
    h = H / ws.N_f
    G = local_to_global_nodes(Nx, Ny)
    F_const = np.zeros(n_patch, dtype=np.complex128)
    np.add.at(F_const, G.ravel(), h * h / 4)
    bub_full = lu.solve(F_const,
                        np.zeros(b_patch.size, dtype=np.complex128))

    P = harm.conj().T @ (B_patch @ harm)
    # Edge trace + linear endpoint subtraction + drop endpoints
    edge_nodes = _edge_trace_indices(t, m, n, ws.N_c, ws.N_f, Nx, Ny)
    leng = ws.N_f + 1
    R_full = harm[edge_nodes, :]
    bub_edge = bub_full[edge_nodes]
    lin_left = np.linspace(1, 0, leng)
    lin_right = np.linspace(0, 1, leng)
    R_full = (R_full
              - np.outer(lin_left, R_full[0, :])
              - np.outer(lin_right, R_full[-1, :]))
    bub_edge = (bub_edge - lin_left * bub_edge[0]
                - lin_right * bub_edge[-1])
    return R_full[1:leng - 1, :], P, bub_edge[1:leng - 1]


# -----------------------------------------------------------------------------
# Edge eigenbasis + per-edge cache
# -----------------------------------------------------------------------------


def _eigen_modes(R, N_mat, P, N_e):
    """Top-N_e eigenvectors of `R^H N R v = λ P v`, ranked by |λ|. Both
    sides are generally complex-Hermitian (no guarantees on definiteness),
    so use the dense general eigenvalue solver and sort by |λ|."""
    M = R.conj().T @ N_mat @ R
    P_reg = P + 1e-12 * np.eye(P.shape[0])
    eigvals, eigvecs = sla.eig(M, P_reg)
    order = np.argsort(-np.abs(eigvals))
    return eigvecs[:, order[:N_e]]


def _build_edge_data(ws, t, m_edge, n_edge, N_e):
    """Per-edge eigen-basis + bubble lifts. Returns
    `(L1·RV, L2·RV, L1·bub, L2·bub, N_e_cached)`; callers slice to the
    requested N_e so larger sweeps don't force a rebuild."""
    L1, L2, N_mat = harmext(ws, m_edge, n_edge, t)
    R, P, bub_edge = restrict(ws, m_edge, n_edge, t)
    V = _eigen_modes(R, N_mat, P, N_e)
    RV = R @ V
    return (L1 @ RV, L2 @ RV, L1 @ bub_edge, L2 @ bub_edge, N_e)


def _cache_needs_rebuild(ws, key, N_e: int) -> bool:
    cached = ws._edge_cache.get(key)
    return cached is None or cached[4] < N_e


def prefactor_edges(ws: Workspace, N_e, n_workers=None):
    """Ensure ws._edge_cache holds ≥ N_e modes per interior edge; edges
    already cached with enough modes are left alone."""
    N_c = ws.N_c
    if ws.param.boundary == "periodic":
        keys = ([(1, m, n) for n in range(N_c) for m in range(N_c)]
                + [(2, m, n) for n in range(N_c) for m in range(N_c)])
    else:
        keys = ([(1, m, n) for n in range(N_c - 1) for m in range(N_c)]
                + [(2, m, n) for n in range(N_c) for m in range(N_c - 1)])
    work = [k for k in keys if _cache_needs_rebuild(ws, k, N_e)]
    if not work:
        return
    with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for key, val in pool.map(lambda k: (k, _build_edge_data(ws, *k, N_e)),
                                  work):
            ws._edge_cache[key] = val


# -----------------------------------------------------------------------------
# Per-cell basis and per-cell bubble (per-step, ψ_n-driven)
# -----------------------------------------------------------------------------


def _nodal_basis_rhs(N_f):
    n_per = 4 * N_f
    f = np.zeros((n_per, 4), dtype=np.complex128)
    f[0:N_f + 1, 0] = np.linspace(1, 0, N_f + 1)
    f[N_f + 1:2 * N_f + 1, 0] = np.linspace(1 - 1 / N_f, 0, N_f)
    f[0:N_f + 1, 1] = np.linspace(0, 1, N_f + 1)
    f[2 * N_f + 1:3 * N_f + 1, 1] = np.linspace(1 - 1 / N_f, 0, N_f)
    f[2 * N_f + 1:3 * N_f + 1, 2] = np.linspace(1 / N_f, 1, N_f)
    f[3 * N_f + 1:4 * N_f, 2] = np.linspace(1 / N_f, 1 - 1 / N_f, N_f - 1)
    f[N_f + 1:2 * N_f + 1, 3] = np.linspace(1 / N_f, 1, N_f)
    f[3 * N_f + 1:4 * N_f, 3] = np.linspace(1 - 1 / N_f, 1 / N_f, N_f - 1)
    return f


def _element_value(ws: Workspace, m, n, N_e):
    """Assemble the (`(N_f+1)² × (4 + 4·(N_e+1))`) complex basis matrix for
    cell (m, n). Standard ExpMsFEM basis: 4 nodal hats + N_e edge eigenmodes
    + 1 edge bubble per edge."""
    N_c, N_f = ws.N_c, ws.N_f
    count = 4 + 4 * (N_e + 1)
    n_nodes = (N_f + 1) ** 2
    value = np.zeros((n_nodes, count), dtype=np.complex128)

    _, _, bdy, lu = ws.cell_factor(m, n)
    F_zero = np.zeros((n_nodes, 4), dtype=np.complex128)
    value[:, 0:4] = lu.solve(F_zero, _nodal_basis_rhs(N_f))

    off = 4
    block = N_e + 1
    periodic = (ws.param.boundary == "periodic")
    # Edge ordering: bottom, top, left, right. Matches the elliptic layout.
    # For periodic BC every edge is interior and neighbour indices wrap mod N_c.
    if periodic or n > 0:
        n_lower = (n - 1) % N_c if periodic else (n - 1)
        L1RV, L2RV, L1b, L2b, _ = ws._edge_cache[(1, m, n_lower)]
        value[:, off:off + N_e] = L2RV[:, :N_e]
        value[:, off + N_e] = L2b
    off += block
    if periodic or n < N_c - 1:
        L1RV, L2RV, L1b, L2b, _ = ws._edge_cache[(1, m, n)]
        value[:, off:off + N_e] = L1RV[:, :N_e]
        value[:, off + N_e] = L1b
    off += block
    if periodic or m > 0:
        m_lower = (m - 1) % N_c if periodic else (m - 1)
        L1RV, L2RV, L1b, L2b, _ = ws._edge_cache[(2, m_lower, n)]
        value[:, off:off + N_e] = L2RV[:, :N_e]
        value[:, off + N_e] = L2b
    off += block
    if periodic or m < N_c - 1:
        L1RV, L2RV, L1b, L2b, _ = ws._edge_cache[(2, m, n)]
        value[:, off:off + N_e] = L1RV[:, :N_e]
        value[:, off + N_e] = L1b
    return value


def _cell_bubble(ws: Workspace, m, n, f_fine_local: np.ndarray) -> np.ndarray:
    """Solve `B_cell · u = f_fine_local` with `u = 0` on the cell perimeter.
    `f_fine_local` has length `(N_f+1)²` and represents the cell-restricted
    fine load at the current time step."""
    _, _, bdy, lu = ws.cell_factor(m, n)
    F = np.asarray(f_fine_local, dtype=np.complex128)
    return lu.solve(F, np.zeros(bdy.size, dtype=np.complex128))


# -----------------------------------------------------------------------------
# ExpMsFEM propagator: offline build + per-step online advance
# -----------------------------------------------------------------------------


def cell_global_indices_periodic(m: int, n: int, N_c: int, N_e: int,
                                  n_per_edge: int) -> np.ndarray:
    """Periodic coarse DOF layout for cell (m, n). Every face of the torus
    is interior; opposite-face DOFs share the same global index.

    Layout:
      * 0 .. N_c²-1                   nodal corners (N_c × N_c grid, wrapped).
      * N_c² .. N_c² + N_c²·n_per_edge − 1  horizontal edges (N_c × N_c grid).
      * rest                          vertical edges (N_c × N_c grid).
    """
    count = 4 + 4 * n_per_edge
    out = np.empty(count, dtype=np.int64)

    def _nodal(i, j):
        return (j % N_c) * N_c + (i % N_c)

    out[0] = _nodal(m, n)
    out[1] = _nodal(m + 1, n)
    out[2] = _nodal(m + 1, n + 1)
    out[3] = _nodal(m, n + 1)

    n_nodal = N_c * N_c
    # Horizontal edges — (col m, row k), flat idx = (k%N_c)*N_c + (m%N_c)
    def _h(col, row):
        return (row % N_c) * N_c + (col % N_c)
    out[4 : 4 + n_per_edge] = (
        n_nodal + _h(m, n) * n_per_edge + np.arange(n_per_edge)
    )
    out[4 + n_per_edge : 4 + 2 * n_per_edge] = (
        n_nodal + _h(m, n + 1) * n_per_edge + np.arange(n_per_edge)
    )

    base_v = n_nodal + N_c * N_c * n_per_edge
    # Vertical edges — (col k, row n), flat idx = (k%N_c)*N_c + (n%N_c)
    def _v(col, row):
        return (col % N_c) * N_c + (row % N_c)
    out[4 + 2 * n_per_edge : 4 + 3 * n_per_edge] = (
        base_v + _v(m, n) * n_per_edge + np.arange(n_per_edge)
    )
    out[4 + 3 * n_per_edge : 4 + 4 * n_per_edge] = (
        base_v + _v(m + 1, n) * n_per_edge + np.arange(n_per_edge)
    )
    return out


def n_total_periodic(N_c: int, n_per_edge: int) -> int:
    """Total coarse DOFs for the periodic layout: nodal + 2 edge blocks."""
    return N_c * N_c * (1 + 2 * n_per_edge)


class SchrodingerPropagator:
    """Amortise the offline cost (Workspace prefactor, edge eigenbasis,
    per-cell basis, coarse operator + LU) and expose a cheap `step(psi_n)`
    per backward-Euler advance."""

    def __init__(self, param: SemiclassicalParam, N_c: int, N_f: int,
                 N_e: int, n_workers: int | None = None,
                 verbose: bool = False):
        self.param = param
        self.N_c, self.N_f, self.N_e = N_c, N_f, N_e
        self.ws = Workspace(param, N_c, N_f)
        t0 = time.time()
        self.ws.prefactor_all(n_workers=n_workers)
        if verbose:
            print(f"  prefactor LUs: {time.time() - t0:.2f}s")
        t0 = time.time()
        prefactor_edges(self.ws, N_e, n_workers=n_workers)
        if verbose:
            print(f"  prefactor edges: {time.time() - t0:.2f}s")

        # Per-cell basis values
        n_cells = N_c * N_c
        count = 4 + 4 * (N_e + 1)
        values = np.empty((n_cells, (N_f + 1) ** 2, count),
                          dtype=np.complex128)

        def _one(job):
            cell, m, n = job
            return cell, _element_value(self.ws, m, n, N_e)

        jobs = [(n * N_c + m, m, n) for n in range(N_c) for m in range(N_c)]
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
            for cell, value in pool.map(_one, jobs):
                values[cell] = value
        if verbose:
            print(f"  per-cell bases: {time.time() - t0:.2f}s")
        self.values = values

        # Coarse B and M via standard assembly. For Dirichlet BC we reuse
        # the elliptic DOF layout (n_per_edge = N_e + 1: N_e edge eigenmodes
        # + 1 edge bubble). For periodic BC we use a reduced periodic
        # layout: N_c² nodal corners + 2·N_c² edge blocks (no "+1 row"
        # of horizontal edges, no "+1 column" of verticals — opposite
        # faces share the same DOF).
        periodic = (param.boundary == "periodic")
        self.periodic = periodic
        self.cell_idx = np.empty((n_cells, count), dtype=np.int64)
        for n in range(N_c):
            for m in range(N_c):
                cell = n * N_c + m
                if periodic:
                    self.cell_idx[cell] = cell_global_indices_periodic(
                        m, n, N_c, N_e, n_per_edge=N_e + 1)
                else:
                    self.cell_idx[cell] = cell_global_indices(
                        m, n, N_c, N_e, n_per_edge=N_e + 1)
        if periodic:
            self.N_total = n_total_periodic(N_c, n_per_edge=N_e + 1)
            self.bdy = np.empty(0, dtype=np.int64)
        else:
            self.N_total = n_total(N_c, N_e, n_per_edge=N_e + 1)
            self.bdy = domain_boundary_dofs(N_c, N_e, n_per_edge=N_e + 1)

        # Build coarse B and M via the complex-symmetric (transpose) pairing:
        #     B_coarse = value^T · B_fine · value,   M_coarse = value^T · M_fine · value.
        # The harmonic-bubble decomposition  ψ = value·c + bubble  is orthogonal
        # in u^T B v (not u^H B v) when B is complex-symmetric, so the
        # transpose pairing gives the decoupled Galerkin system needed here.
        B_elems = np.empty((n_cells, count, count), dtype=np.complex128)
        M_elems = np.empty((n_cells, count, count), dtype=np.complex128)
        for n in range(N_c):
            for m in range(N_c):
                cell = n * N_c + m
                B_c, M_c, _, _ = self.ws.cell_factor(m, n)
                v = values[cell]
                B_elems[cell] = v.T @ (B_c @ v)
                M_elems[cell] = v.T @ (M_c @ v)
        I = np.broadcast_to(self.cell_idx[:, :, None],
                            (n_cells, count, count)).reshape(-1)
        J = np.broadcast_to(self.cell_idx[:, None, :],
                            (n_cells, count, count)).reshape(-1)
        self.B_coarse = sp.coo_matrix(
            (B_elems.reshape(-1), (I, J)),
            shape=(self.N_total, self.N_total)).tocsc()
        self.M_coarse = sp.coo_matrix(
            (M_elems.reshape(-1), (I, J)),
            shape=(self.N_total, self.N_total)).tocsc()

        interior_mask = np.ones(self.N_total, dtype=bool)
        interior_mask[self.bdy] = False
        self.interior = np.where(interior_mask)[0]
        self.B_coarse_lu = spla.splu(
            self.B_coarse[self.interior][:, self.interior].tocsc())

        self.count = count
        self.N_fine_side = N_c * N_f

    # ----- projection / reconstruction ------------------------------------

    def project_initial(self, psi0_fine: np.ndarray) -> np.ndarray:
        """L²-project a fine-scale initial condition onto the coarse basis.

        Returns the coarse coefficient vector `c` such that `value · c`
        best approximates `psi0_fine` in the L² (mass-matrix) norm.
        """
        # RHS: value.T · M_fine · psi0, accumulated per cell.
        rhs = np.zeros(self.N_total, dtype=np.complex128)
        N_c, N_f = self.N_c, self.N_f
        psi_grid = psi0_fine.reshape(self.N_fine_side + 1,
                                     self.N_fine_side + 1)
        for n in range(N_c):
            for m in range(N_c):
                cell = n * N_c + m
                _, M_c, _, _ = self.ws.cell_factor(m, n)
                psi_cell = psi_grid[n * N_f : n * N_f + N_f + 1,
                                     m * N_f : m * N_f + N_f + 1].ravel()
                # Transpose pairing matching the coarse M assembly convention.
                contrib = self.values[cell].T @ (M_c @ psi_cell)
                rhs[self.cell_idx[cell]] += contrib
        # Solve (M_coarse on interior) · c_int = rhs_int with bdy = 0.
        rhs_int = rhs[self.interior]
        M_int = self.M_coarse[self.interior][:, self.interior].tocsc()
        c_int = spla.spsolve(M_int, rhs_int)
        c = np.zeros(self.N_total, dtype=np.complex128)
        c[self.interior] = c_int
        return c

    def reconstruct_fine(self, c: np.ndarray) -> np.ndarray:
        """Return `ψ_fine = value · c` on the full (N_c·N_f+1)² grid."""
        N_c, N_f = self.N_c, self.N_f
        out = np.zeros((self.N_fine_side + 1, self.N_fine_side + 1),
                       dtype=np.complex128)
        for n in range(N_c):
            for m in range(N_c):
                cell = n * N_c + m
                coef = c[self.cell_idx[cell]]
                zs = self.values[cell] @ coef
                out[n * N_f : n * N_f + N_f + 1,
                    m * N_f : m * N_f + N_f + 1] = zs.reshape(N_f + 1, N_f + 1)
        return out.ravel()

    # ----- online per-step advance ----------------------------------------

    def step(self, psi_n_fine: np.ndarray):
        """Advance one backward-Euler step from fine-scale `psi_n`. Returns
        `(psi_{n+1}_fine, c_{n+1}, bubble_fine)`.

        Uses the harmonic-bubble decomposition
            ψ_{n+1} = value · c_{n+1} + ψ_B
        where the cell bubble ψ_B solves `B_cell · ψ_B = shift · M · ψ_n`
        (Dirichlet-0 on cell perimeter) and the coarse harmonic coefficients
        satisfy
            B_coarse · c_{n+1} = shift · (value^H · M · ψ_n).
        """
        shift = self.param.im_shift  # -iε/Δt
        N_c, N_f = self.N_c, self.N_f
        psi_grid = psi_n_fine.reshape(self.N_fine_side + 1,
                                      self.N_fine_side + 1)

        rhs_coarse = np.zeros(self.N_total, dtype=np.complex128)
        bubble_grid = np.zeros(psi_grid.shape, dtype=np.complex128)
        for n in range(N_c):
            for m in range(N_c):
                cell = n * N_c + m
                B_c, M_c, _, _ = self.ws.cell_factor(m, n)
                psi_cell = psi_grid[n * N_f : n * N_f + N_f + 1,
                                     m * N_f : m * N_f + N_f + 1].ravel()
                f_cell = shift * (M_c @ psi_cell)

                # Cell bubble at step n+1
                u_B = _cell_bubble(self.ws, m, n, f_cell)

                # Coarse RHS: value^T · f_cell (transpose pairing —
                # matches the coarse B_coarse = value^T · B · value so the
                # harmonic-bubble decoupling holds exactly).
                contrib = self.values[cell].T @ f_cell
                rhs_coarse[self.cell_idx[cell]] += contrib

                bubble_grid[n * N_f : n * N_f + N_f + 1,
                             m * N_f : m * N_f + N_f + 1] += \
                    u_B.reshape(N_f + 1, N_f + 1)

        rhs_int = rhs_coarse[self.interior]
        c_new_int = self.B_coarse_lu.solve(rhs_int)
        c_new = np.zeros(self.N_total, dtype=np.complex128)
        c_new[self.interior] = c_new_int

        psi_new_fine = self.reconstruct_fine(c_new) + bubble_grid.ravel()
        return psi_new_fine, c_new, bubble_grid.ravel()


def run_expmsfem_schrodinger(param: SemiclassicalParam, psi0_fine: np.ndarray,
                              N_c: int, N_f: int, N_e: int, n_steps: int,
                              save_stride: int = 1,
                              n_workers: int | None = None,
                              verbose: bool = False):
    """End-to-end wrapper: offline build + all `n_steps` backward-Euler
    advances. The initial ψ₀ is kept in the fine representation (no
    coarse projection) and each step's RHS uses the previous step's
    fine ψ_n directly. Returns `(ts, frames, propagator)` with the
    same shape convention as `solve_fine_backward_euler`."""
    prop = SchrodingerPropagator(param, N_c, N_f, N_e,
                                 n_workers=n_workers, verbose=verbose)
    psi_fine = np.asarray(psi0_fine, dtype=np.complex128).copy()
    ts = [0.0]
    frames = [psi_fine.copy()]
    for step in range(1, n_steps + 1):
        psi_fine, _, _ = prop.step(psi_fine)
        if step % save_stride == 0:
            ts.append(step * param.dt)
            frames.append(psi_fine.copy())
    return np.array(ts), np.array(frames).T, prop
