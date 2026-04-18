"""Per-coarse-cell ExpMsFEM basis construction.

Mirrors Matlab elementstiff.m for the elliptic periodic case. For each coarse
cell (m, n) in 0-indexed coordinates (0 <= m, n < N_c), produces the full
fine-mesh basis `value` of shape ((N_f+1)^2, count) where

    count = 4 + 4*(N_e + 1)

(4 nodal + 4 edges × (N_e + 1) modes each). Also returns the element stiffness
K_elem = value' B value and the element load f_elem, where B is the cell's
Q1 fine-stiffness (from local_ops.basefun).

Edge ordering in the DOF list: bottom, top, left, right. Boundary edges (touching
∂Ω) contribute zero columns.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from . import local_ops
from .mesh import cell_boundary_nodes, local_to_global_nodes


def _nodal_basis_rhs(N_f: int) -> np.ndarray:
    """Boundary-data matrix f of shape (4*N_f, 4) for the four nodal hats."""
    n_per = 4 * N_f
    f = np.zeros((n_per, 4), dtype=np.float64)

    # Corner 1 = (0,0).
    f[0:N_f + 1, 0] = np.linspace(1, 0, N_f + 1)
    f[N_f + 1:2 * N_f + 1, 0] = np.linspace(1 - 1 / N_f, 0, N_f)

    # Corner 2 = (N_f, 0).
    f[0:N_f + 1, 1] = np.linspace(0, 1, N_f + 1)
    f[2 * N_f + 1:3 * N_f + 1, 1] = np.linspace(1 - 1 / N_f, 0, N_f)

    # Corner 3 = (N_f, N_f).
    f[2 * N_f + 1:3 * N_f + 1, 2] = np.linspace(1 / N_f, 1, N_f)
    f[3 * N_f + 1:4 * N_f, 2] = np.linspace(1 / N_f, 1 - 1 / N_f, N_f - 1)

    # Corner 4 = (0, N_f).
    f[N_f + 1:2 * N_f + 1, 3] = np.linspace(1 / N_f, 1, N_f)
    f[3 * N_f + 1:4 * N_f, 3] = np.linspace(1 - 1 / N_f, 1 / N_f, N_f - 1)
    return f


def _nodal_basis(B: sp.csc_matrix, N_f: int) -> np.ndarray:
    """Classical MsFEM nodal basis on a coarse cell given the cell fine
    stiffness B. Returns value shape ((N_f+1)^2, 4)."""
    f = _nodal_basis_rhs(N_f)
    bdy = cell_boundary_nodes(N_f, N_f)
    n_nodes = (N_f + 1) ** 2
    F_zero = np.zeros((n_nodes, 4), dtype=np.float64)
    return local_ops._interior_solve(B, F_zero, bdy, f)


def _nodal_basis_cached(ws: "local_ops.Workspace", m: int, n: int) -> np.ndarray:
    B, bdy, lu = ws.cell_factor(m, n)
    f = _nodal_basis_rhs(ws.N_f)
    n_nodes = (ws.N_f + 1) ** 2
    F_zero = np.zeros((n_nodes, 4), dtype=np.float64)
    return lu.solve(F_zero, bdy, f)


def _eigen_modes(R: np.ndarray, N: np.ndarray, P: np.ndarray, N_e: int) -> np.ndarray:
    """Top-N_e generalized eigenvectors of (R'NR v = λ P v).

    Uses LAPACK's `?sygvx` via scipy's `subset_by_index` so the cost scales
    with N_e (typically 1..10) rather than n_active (~100-300).

    Note: R has N_f-1 rows, so R'NR has rank ≤ N_f-1. If N_e ≤ N_f-1 (which it
    always is in practice), subset_by_index=[n-N_e, n-1] gives the meaningful
    modes.
    """
    M = R.T @ N @ R
    M = 0.5 * (M + M.T)
    P_reg = P + 1e-14 * np.eye(P.shape[0])
    P_reg = 0.5 * (P_reg + P_reg.T)
    n = M.shape[0]
    # eigh returns ascending; request the top N_e by absolute index.
    lo = max(0, n - N_e)
    hi = n - 1
    eigvals, eigvecs = sla.eigh(M, P_reg, subset_by_index=[lo, hi])
    # Reorder to descending (largest first) for downstream consistency
    return eigvecs[:, ::-1]


def _build_edge_data(ws, t, m_edge, n_edge, N_e):
    """Compute the per-edge data shared by both adjacent cells: the lifted
    eigen-basis on each side and the lifted edge-bubble on each side. Returns
    a 5-tuple (L1_RV, L2_RV, L1_bub, L2_bub, N_e_cached) with N_e_cached == N_e.
    The first four arrays have N_e_cached columns; callers slice to the
    requested count (see `_edge_contribution_cached`) to reuse across N_e.
    """
    L1, L2, N = local_ops.harmext_cached(ws, m_edge, n_edge, t)
    R, P, bub = local_ops.restrict_cached(ws, m_edge, n_edge, t)
    V = _eigen_modes(R, N, P, N_e)
    RV = R @ V
    return (L1 @ RV, L2 @ RV, L1 @ bub, L2 @ bub, N_e)


def prefactor_edges(ws, N_e: int, n_workers: int | None = None):
    """Ensure ws._edge_cache holds ≥ N_e eigen-modes for every interior edge.

    Called once after ws.prefactor_all(). Each interior edge is computed only
    if the cache is empty or holds fewer than N_e modes, so sweeps over
    increasing N_e reuse prior work. Edges are computed in parallel via a
    thread pool.
    """
    import concurrent.futures as cf

    N_c = ws.N_c
    horiz = [(1, m, n) for n in range(N_c - 1) for m in range(N_c)]
    vert = [(2, m, n) for n in range(N_c) for m in range(N_c - 1)]
    keys = horiz + vert
    work = [k for k in keys if _cache_needs_rebuild(ws, k, N_e)]
    if not work:
        return

    def _one(key):
        t, m_edge, n_edge = key
        return key, _build_edge_data(ws, t, m_edge, n_edge, N_e)

    with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for key, val in pool.map(_one, work):
            ws._edge_cache[key] = val


def _cache_needs_rebuild(ws, key, N_e: int) -> bool:
    cached = ws._edge_cache.get(key)
    return cached is None or cached[4] < N_e


def _edge_contribution_cached(ws, m_edge, n_edge, t, side, N_e):
    """Fast path: lookup pre-built edge data from ws._edge_cache. If not
    present (e.g. prefactor_edges wasn't called) or cached with fewer than
    N_e modes, (re)build on the fly; otherwise slice the leading N_e columns."""
    key = (t, m_edge, n_edge)
    if _cache_needs_rebuild(ws, key, N_e):
        ws._edge_cache[key] = _build_edge_data(ws, t, m_edge, n_edge, N_e)
    L1_RV, L2_RV, L1_bub, L2_bub, _ = ws._edge_cache[key]
    if side == "low":
        return np.concatenate([L1_RV[:, :N_e], L1_bub.reshape(-1, 1)], axis=1)
    else:
        return np.concatenate([L2_RV[:, :N_e], L2_bub.reshape(-1, 1)], axis=1)


def element_basis_cached(ws: "local_ops.Workspace", m: int, n: int, N_e: int):
    """Assemble the full per-cell basis using cached factors."""
    N_c = ws.N_c
    N_f = ws.N_f
    count = 4 + 4 * (N_e + 1)
    n_nodes = (N_f + 1) ** 2
    value = np.zeros((n_nodes, count), dtype=np.float64)

    B, _, _ = ws.cell_factor(m, n)
    value[:, 0:4] = _nodal_basis_cached(ws, m, n)

    off = 4
    block = N_e + 1

    if n > 0:
        value[:, off:off + block] = _edge_contribution_cached(ws, m, n - 1, 1, "high", N_e)
    off += block

    if n < N_c - 1:
        value[:, off:off + block] = _edge_contribution_cached(ws, m, n, 1, "low", N_e)
    off += block

    if m > 0:
        value[:, off:off + block] = _edge_contribution_cached(ws, m - 1, n, 2, "high", N_e)
    off += block

    if m < N_c - 1:
        value[:, off:off + block] = _edge_contribution_cached(ws, m, n, 2, "low", N_e)

    K_elem = value.T @ (B @ value)
    F_node = local_ops._cell_constant_rhs(N_c, N_f)
    f_elem = value.T @ F_node
    return value, B, K_elem, f_elem


# Legacy non-cached entry point for tests/backward compat.
def element_basis(a_fun: Callable, m: int, n: int, N_c: int, N_f: int,
                  N_e: int):
    ws = local_ops.Workspace(a_fun, N_c, N_f)
    return element_basis_cached(ws, m, n, N_e)
