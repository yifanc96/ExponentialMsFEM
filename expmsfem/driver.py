"""End-to-end driver: build all per-cell bases, assemble, solve, reconstruct,
and compute errors. Reproduces Matlab MsFEM.m + main.m pipeline.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import assembly, element_basis, errors, fem, local_ops, reconstruct


def run_expmsfem(a_fun: Callable, N_c: int, N_f: int, N_e: int,
                 u_ref: np.ndarray | None = None,
                 K_ref=None, M_ref=None,
                 verbose: bool = False,
                 n_workers: int | None = None,
                 workspace: "local_ops.Workspace | None" = None):
    """Solve the ExpMsFEM problem on Ω=(0,1)² with coefficient a_fun.

    RHS ≡ -1, homogeneous Dirichlet on ∂Ω, N_e edge modes + 1 edge-bubble per
    interior edge, on a coarse N_c × N_c grid with N_f × N_f fine elements per
    coarse cell.

    Parallelism: offline phase uses a thread pool of `n_workers` threads
    (defaults to the ThreadPoolExecutor default, typically #cores). Pass
    n_workers=1 for serial execution. Pass a pre-built `workspace` (factored
    via `ws.prefactor_all()`) to reuse factorisations across runs.

    Returns a dict with keys {"u_ms_fine", "u_ref_fine", "e_L2", "e_H1",
    "K_ref", "M_ref"}.
    """
    import concurrent.futures as cf
    import time

    count = 4 + 4 * (N_e + 1)
    n_cells = N_c * N_c
    n_fine_nodes = (N_f + 1) ** 2

    # ---- Offline phase 1: factorise every cell + edge patch in parallel ----
    if workspace is None:
        ws = local_ops.Workspace(a_fun, N_c, N_f)
        t0 = time.time()
        ws.prefactor_all(n_workers=n_workers)
        if verbose:
            print(f"  prefactor LUs: {time.time() - t0:.2f}s "
                  f"({len(ws._cell)} cells, {len(ws._patch)} patches)")
    else:
        ws = workspace
        # Edge cache is reused across N_e values: `prefactor_edges` enlarges
        # each edge's eigen-mode set only when the request exceeds what is
        # already cached.

    # ---- Offline phase 2: per-edge basis (harmext + restrict + eigh) ----
    t0 = time.time()
    element_basis.prefactor_edges(ws, N_e, n_workers=n_workers)
    if verbose:
        print(f"  prefactor edges: {time.time() - t0:.2f}s "
              f"({len(ws._edge_cache)} edges)")

    # ---- Offline phase 3: per-cell basis assembly in parallel ----
    values = np.empty((n_cells, n_fine_nodes, count), dtype=np.float64)
    K_elems = np.empty((n_cells, count, count), dtype=np.float64)
    f_elems = np.empty((n_cells, count), dtype=np.float64)
    bubbles = np.empty((n_cells, n_fine_nodes), dtype=np.float64)

    def _one_cell(idx_mn):
        cell, m, n = idx_mn
        value, B, K_elem, f_elem = element_basis.element_basis_cached(ws, m, n, N_e)
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

    # ---- Global assembly ----
    cell_idx = assembly.all_cell_global_indices(N_c, N_e)
    N_total = assembly.n_total(N_c, N_e)
    A_glob, F_glob = assembly.assemble_global(K_elems, f_elems, cell_idx, N_total)
    bdy = assembly.domain_boundary_dofs(N_c, N_e)
    u_global = assembly.solve_coarse(A_glob, F_glob, bdy)

    # ---- Reconstruction ----
    u_ms_fine = reconstruct.reconstruct_fine(values, u_global, bubbles, N_c, N_f, N_e)

    # ---- Error vs fine reference ----
    if u_ref is None or K_ref is None or M_ref is None:
        N_total_fine = N_c * N_f
        u_ref, K_ref, M_ref, _ = fem.solve_fine(a_fun, N_total_fine)

    e_L2 = errors.rel_l2(u_ref, u_ms_fine, M_ref)
    e_H1 = errors.rel_h1(u_ref, u_ms_fine, K_ref)
    return {
        "u_ms_fine": u_ms_fine,
        "u_ref_fine": u_ref,
        "e_L2": e_L2,
        "e_H1": e_H1,
        "K_ref": K_ref,
        "M_ref": M_ref,
    }
