"""Baseline multiscale methods for apples-to-apples comparison with the main
ExpMsFEM pipeline.

Two variants, both mirroring the corresponding Matlab directories:

  * `H+bubble` (Matlab `elliptic/*/H+bubble/`): N_e edge eigenmodes per
    interior edge + 1 interior bubble per coarse cell (added during fine
    reconstruction). No dedicated "edge-bubble" DOF column per edge.

  * `O(H)` (Matlab `elliptic/*/O(H)/`): N_e edge eigenmodes per interior edge,
    NO interior cell bubble. Pure coarse-scale nodal + edge basis.

Both share the existing Workspace / harmext / restrict / eigenproblem
infrastructure with the Exp pipeline; they just drop (a) the per-edge "bubble"
column from each cell's basis and (b) optionally the cell-bubble addition in
reconstruction.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from typing import Callable

import numpy as np

from . import assembly, element_basis, fem, local_ops
from .element_basis import _build_edge_data, _cache_needs_rebuild, _nodal_basis_cached


def _edge_contribution_nobub(ws: local_ops.Workspace, m_edge: int, n_edge: int,
                             t: int, side: str, N_e: int) -> np.ndarray:
    """Return the leading N_e basis columns (no edge-bubble) for one side."""
    key = (t, m_edge, n_edge)
    if _cache_needs_rebuild(ws, key, N_e):
        ws._edge_cache[key] = _build_edge_data(ws, t, m_edge, n_edge, N_e)
    L1_RV, L2_RV, _, _, _ = ws._edge_cache[key]
    return (L1_RV if side == "low" else L2_RV)[:, :N_e]


def _element_basis_baseline(ws: local_ops.Workspace, m: int, n: int, N_e: int):
    """Per-cell basis without the edge-bubble column. count = 4 + 4·N_e."""
    N_c, N_f = ws.N_c, ws.N_f
    count = 4 + 4 * N_e
    n_nodes = (N_f + 1) ** 2
    value = np.zeros((n_nodes, count), dtype=np.float64)

    B, _, _ = ws.cell_factor(m, n)
    value[:, 0:4] = _nodal_basis_cached(ws, m, n)

    off = 4
    if n > 0 and N_e > 0:
        value[:, off:off + N_e] = _edge_contribution_nobub(ws, m, n - 1, 1, "high", N_e)
    off += N_e
    if n < N_c - 1 and N_e > 0:
        value[:, off:off + N_e] = _edge_contribution_nobub(ws, m, n, 1, "low", N_e)
    off += N_e
    if m > 0 and N_e > 0:
        value[:, off:off + N_e] = _edge_contribution_nobub(ws, m - 1, n, 2, "high", N_e)
    off += N_e
    if m < N_c - 1 and N_e > 0:
        value[:, off:off + N_e] = _edge_contribution_nobub(ws, m, n, 2, "low", N_e)

    K_elem = value.T @ (B @ value)
    F_node = local_ops._cell_constant_rhs(N_c, N_f)
    f_elem = value.T @ F_node
    return value, B, K_elem, f_elem


def run_baseline(a_fun: Callable, N_c: int, N_f: int, N_e: int,
                 *,
                 include_cell_bubble: bool,
                 u_ref: np.ndarray | None = None,
                 K_ref=None, M_ref=None,
                 workspace: "local_ops.Workspace | None" = None,
                 n_workers: int | None = None,
                 verbose: bool = False):
    """Shared driver for H+bubble (`include_cell_bubble=True`) and O(H)
    (`include_cell_bubble=False`).
    """
    count = 4 + 4 * N_e
    n_cells = N_c * N_c
    n_fine_nodes = (N_f + 1) ** 2

    if workspace is None:
        ws = local_ops.Workspace(a_fun, N_c, N_f)
        t0 = time.time()
        ws.prefactor_all(n_workers=n_workers)
        if verbose:
            print(f"  prefactor LUs: {time.time() - t0:.2f}s")
    else:
        ws = workspace
        # Reuse edge cache: prefactor_edges only rebuilds edges with fewer
        # than N_e cached modes.

    # Prefactor edges (populates ws._edge_cache) using the SAME routine as Exp
    t0 = time.time()
    element_basis.prefactor_edges(ws, N_e, n_workers=n_workers)
    if verbose:
        print(f"  prefactor edges: {time.time() - t0:.2f}s")

    values = np.empty((n_cells, n_fine_nodes, count), dtype=np.float64)
    K_elems = np.empty((n_cells, count, count), dtype=np.float64)
    f_elems = np.empty((n_cells, count), dtype=np.float64)
    bubbles = np.empty((n_cells, n_fine_nodes), dtype=np.float64)

    def _one_cell(job):
        cell, m, n = job
        value, _, K_elem, f_elem = _element_basis_baseline(ws, m, n, N_e)
        bub = local_ops.bubble_cached(ws, m, n)
        return cell, value, K_elem, f_elem, bub

    jobs = [(n * N_c + m, m, n) for n in range(N_c) for m in range(N_c)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for cell, value, K_elem, f_elem, bub in pool.map(_one_cell, jobs):
            values[cell] = value
            K_elems[cell] = K_elem
            f_elems[cell] = f_elem
            bubbles[cell] = bub
    if verbose:
        print(f"  per-cell bases: {time.time() - t0:.2f}s")

    # Global assembly with baseline DOF layout (n_per_edge = N_e, no +1)
    n_per_edge = N_e
    cell_idx = assembly.all_cell_global_indices(N_c, N_e, n_per_edge=n_per_edge)
    N_total = assembly.n_total(N_c, N_e, n_per_edge=n_per_edge)
    A_glob, F_glob = assembly.assemble_global(K_elems, f_elems, cell_idx, N_total)
    bdy = assembly.domain_boundary_dofs(N_c, N_e, n_per_edge=n_per_edge)
    u_global = assembly.solve_coarse(A_glob, F_glob, bdy)

    # Reconstruct fine solution
    N = N_c * N_f
    fine = np.zeros((N + 1, N + 1), dtype=np.float64)
    for n in range(N_c):
        for m in range(N_c):
            cell = n * N_c + m
            gidx = assembly.cell_global_indices(m, n, N_c, N_e, n_per_edge=n_per_edge)
            zs = values[cell] @ u_global[gidx]
            if include_cell_bubble:
                zs = zs + bubbles[cell]
            zs_2d = zs.reshape(N_f + 1, N_f + 1)
            fine[n * N_f : n * N_f + N_f + 1,
                 m * N_f : m * N_f + N_f + 1] = zs_2d
    u_ms_fine = fine.ravel()

    if u_ref is None or K_ref is None or M_ref is None:
        u_ref, K_ref, M_ref, _ = fem.solve_fine(a_fun, N_c * N_f)

    e = u_ref - u_ms_fine
    num_L2 = float(e @ (M_ref @ e))
    den_L2 = float(u_ref @ (M_ref @ u_ref))
    e_L2 = np.sqrt(num_L2 / den_L2) if den_L2 > 0 else np.inf
    num_H1 = float(e @ (K_ref @ e))
    den_H1 = float(u_ref @ (K_ref @ u_ref))
    e_H1 = np.sqrt(num_H1 / den_H1) if den_H1 > 0 else np.inf

    return {
        "u_ms_fine": u_ms_fine,
        "u_ref_fine": u_ref,
        "e_L2": e_L2,
        "e_H1": e_H1,
        "K_ref": K_ref,
        "M_ref": M_ref,
    }


def run_hbubble(a_fun: Callable, N_c: int, N_f: int, N_e: int, **kw):
    """Classical MsFEM + bubble (Matlab H+bubble): N_e edge eigenmodes per
    interior edge, cell bubble added during reconstruction."""
    return run_baseline(a_fun, N_c, N_f, N_e, include_cell_bubble=True, **kw)


def run_OH(a_fun: Callable, N_c: int, N_f: int, N_e: int, **kw):
    """Classical MsFEM (Matlab O(H)): N_e edge eigenmodes per edge, NO cell
    bubble in reconstruction."""
    return run_baseline(a_fun, N_c, N_f, N_e, include_cell_bubble=False, **kw)
