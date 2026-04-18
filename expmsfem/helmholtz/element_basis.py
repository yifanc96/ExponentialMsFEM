"""Helmholtz per-cell basis: nodal + edge-eigenmode basis, sesquilinear
element stiffness, and load with boundary impedance corrections.

For interior cells this mirrors the elliptic construction (4 corner hats +
4 edges × (N_e+1) modes). For cells touching ∂Ω, the corner nodal basis uses
Dirichlet data only on the cell sides that are INTERIOR to the coarse grid —
the domain-boundary sides stay free and are governed by the impedance term in
the local Helmholtz operator. This matches Matlab helmholtz/case1/Exp/
elementstiff.m's nine-branch construction, but we implement it as a single
branch by deriving the boundary data analytically from the Q1 hat formulas.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from . import coefficients as hc
from . import local_ops as hlo
from . import q1 as hq1


def _corner_hat_values_at_indices(N_f: int, node_indices: np.ndarray) -> np.ndarray:
    """Boundary values for each of 4 Q1 corner hats at the given local node
    indices. Returns array of shape (len(node_indices), 4), row k = values
    of the 4 corner hats evaluated at node_indices[k].
    """
    Nxp1 = N_f + 1
    j = node_indices // Nxp1
    i = node_indices - j * Nxp1
    u = i / N_f
    v = j / N_f
    f = np.zeros((node_indices.size, 4), dtype=np.complex128)
    f[:, 0] = (1 - u) * (1 - v)
    f[:, 1] = u * (1 - v)
    f[:, 2] = u * v
    f[:, 3] = (1 - u) * v
    return f


def nodal_basis(ws: hlo.HelmWorkspace, m: int, n: int) -> np.ndarray:
    """Compute the 4 nodal-basis columns for cell (m, n) of shape
    ((N_f+1)², 4). For interior cells these reduce to the standard MsFEM
    nodal basis; for boundary cells, the ∂Ω sides are free (Robin).
    """
    N_f = ws.N_f
    A, bdy, lu, lu_T = ws.cell_factor(m, n)
    # Dirichlet data on the bdy nodes = values of the 4 corner hats there.
    hats_on_bdy = _corner_hat_values_at_indices(N_f, bdy)
    # Solve A u = 0 with u[bdy] = hats_on_bdy.
    F_zero = np.zeros(((N_f + 1) ** 2, 4), dtype=np.complex128)
    return lu.solve(F_zero, bdy, hats_on_bdy)


def _eigen_modes(R: np.ndarray, N: np.ndarray, P: np.ndarray, N_e: int) -> np.ndarray:
    """Top-N_e generalised eigenvectors of  R' N R v = λ P v, complex general.

    In the Helmholtz case neither M = R' N R nor P is Hermitian (both use
    complex-symmetric / mixed-conjugate projections because the underlying
    stiffness K is complex symmetric — not Hermitian — due to the impedance
    term). We use scipy's general complex `eig` solver and keep the top
    |λ|-largest modes, matching Matlab's default `eigs(M, P, N_e)` behaviour.

    R' N R has rank ≤ R.shape[0] (the number of edge-interior DOFs), so we
    clamp N_e to that rank and pad with zero columns if asked for more.
    """
    M = R.conj().T @ N @ R
    P_reg = P + 1e-12 * np.eye(P.shape[0])

    # Dense complex general eigendecomposition. R' N R has rank ≤ R.shape[0]
    # (the number of edge-interior DOFs). The top modes by |λ| are well-
    # behaved up to rank - 1; the rank-th mode sits right at the boundary
    # between meaningful and null-space and is numerically unreliable. Users
    # should therefore keep N_e ≤ R.shape[0] − 1 for stable results.
    eigvals, eigvecs = sla.eig(M, P_reg)
    order = np.argsort(-np.abs(eigvals))
    k_eff = min(N_e, order.size)
    V = eigvecs[:, order[:k_eff]]
    if k_eff < N_e:
        pad = np.zeros((V.shape[0], N_e - k_eff), dtype=V.dtype)
        V = np.concatenate([V, pad], axis=1)
    return V


def _build_edge_data(ws: hlo.HelmWorkspace, t: int, m_edge: int, n_edge: int,
                     N_e: int):
    """Per-interior-edge computation: harmext + restrict + eigen selection,
    return (L1·R·V, L2·R·V, L1·bub, L2·bub, N_e_cached). Caller slices the
    leading N_e columns to reuse the cache across increasing N_e."""
    L1, L2, N = hlo.harmext(ws, m_edge, n_edge, t)
    R, P, bub = hlo.restrict(ws, m_edge, n_edge, t)
    V = _eigen_modes(R, N, P, N_e)
    RV = R @ V
    return (L1 @ RV, L2 @ RV, L1 @ bub, L2 @ bub, N_e)


def _cache_needs_rebuild(ws: hlo.HelmWorkspace, key, N_e: int) -> bool:
    cached = ws._edge_cache.get(key)
    return cached is None or cached[4] < N_e


def prefactor_edges(ws: hlo.HelmWorkspace, N_e: int,
                    n_workers: int | None = None):
    """Ensure ws._edge_cache has ≥ N_e eigen-modes for every interior edge.
    Already-cached edges with enough modes are left untouched."""
    import concurrent.futures as cf

    N_c = ws.N_c
    keys = (
        [(1, m, n) for n in range(N_c - 1) for m in range(N_c)]
        + [(2, m, n) for n in range(N_c) for m in range(N_c - 1)]
    )
    work = [k for k in keys if _cache_needs_rebuild(ws, k, N_e)]
    if not work:
        return

    def _one(key):
        t, m, n = key
        return key, _build_edge_data(ws, t, m, n, N_e)

    with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for key, val in pool.map(_one, work):
            ws._edge_cache[key] = val


def _edge_contribution(ws: hlo.HelmWorkspace, m_edge: int, n_edge: int,
                       t: int, side: str, N_e: int) -> np.ndarray:
    key = (t, m_edge, n_edge)
    if _cache_needs_rebuild(ws, key, N_e):
        ws._edge_cache[key] = _build_edge_data(ws, t, m_edge, n_edge, N_e)
    L1_RV, L2_RV, L1_bub, L2_bub, _ = ws._edge_cache[key]
    if side == "low":
        return np.concatenate([L1_RV[:, :N_e], L1_bub.reshape(-1, 1)], axis=1)
    return np.concatenate([L2_RV[:, :N_e], L2_bub.reshape(-1, 1)], axis=1)


def element_basis(ws: hlo.HelmWorkspace, m: int, n: int, N_e: int):
    """Full (N_f+1)² × count Helmholtz basis + element stiffness + load.

    Returns (value, B_norm, K_elem, f_elem) where
      value : complex basis matrix
      B_norm : norm reference matrix for the energy norm (real-positive)
      K_elem : complex sesquilinear element stiffness value.T @ A @ conj(value)
      f_elem : complex element load (see below)
    """
    N_c, N_f = ws.N_c, ws.N_f
    count = 4 + 4 * (N_e + 1)
    n_nodes = (N_f + 1) ** 2

    value = np.zeros((n_nodes, count), dtype=np.complex128)
    # Nodal basis (4)
    value[:, 0:4] = nodal_basis(ws, m, n)

    block = N_e + 1
    off = 4
    if n > 0:
        value[:, off:off + block] = _edge_contribution(ws, m, n - 1, 1, "high", N_e)
    off += block
    if n < N_c - 1:
        value[:, off:off + block] = _edge_contribution(ws, m, n, 1, "low", N_e)
    off += block
    if m > 0:
        value[:, off:off + block] = _edge_contribution(ws, m - 1, n, 2, "high", N_e)
    off += block
    if m < N_c - 1:
        value[:, off:off + block] = _edge_contribution(ws, m, n, 2, "low", N_e)

    A, _, _, _ = ws.cell_factor(m, n)
    # Element stiffness: sesquilinear form  K_elem = value.T · A · conj(value)
    K_elem = value.T @ (A @ value.conj())

    # Element load: mirrors Matlab elementstiff.m
    #   F(cell-corner) = ffun·h²/4 · conj(value[corner, :])  + gfun boundary terms
    # Assemble the full fine-scale load vector once and project via conj(value).
    xs = np.linspace(m / N_c, (m + 1) / N_c, N_f + 1)
    ys = np.linspace(n / N_c, (n + 1) / N_c, N_f + 1)
    F_fine = hlo._assemble_helm_load(xs, ys, ws.k0)
    f_elem = value.conj().T @ F_fine

    # Matlab elementstiff.m subtracts `bubble.' * B * conj(value)` from f.
    # That is, the bubble particular solution is built into the load vector
    # so the coarse DOFs solve for the "non-bubble" part only. Add the
    # correction:
    bub_fine = hlo.bubble(m, n, N_c, N_f, ws.k0)
    f_elem = f_elem - bub_fine @ (A @ value.conj())

    return value, A, K_elem, f_elem
