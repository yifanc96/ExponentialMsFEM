"""End-to-end Helmholtz ExpMsFEM driver."""

from __future__ import annotations

import time
import concurrent.futures as cf

import numpy as np

from ..assembly import all_cell_global_indices, n_total
from ..mesh import local_to_global_nodes
from . import assembly as hass
from . import element_basis as heb
from . import fem as hfem
from . import local_ops as hlo


def reconstruct_fine(values, u_global, bubbles, N_c, N_f, N_e):
    """Paste the per-cell basis·coefs + bubble into a global fine array.
    Matches the elliptic reconstruction but with complex dtype and the
    Matlab MsFEM convention that the bubble is added on top of each cell's
    projection."""
    from ..assembly import cell_global_indices

    N = N_c * N_f
    fine = np.zeros((N + 1, N + 1), dtype=np.complex128)
    for n in range(N_c):
        for m in range(N_c):
            cell = n * N_c + m
            gidx = cell_global_indices(m, n, N_c, N_e)
            zs = values[cell] @ u_global[gidx]
            zs = zs + bubbles[cell]
            zs_2d = zs.reshape(N_f + 1, N_f + 1)
            fine[n * N_f : n * N_f + N_f + 1,
                 m * N_f : m * N_f + N_f + 1] = zs_2d
    return fine.ravel()


def run_expmsfem_helm(N_c: int, N_f: int, N_e: int, k0: float,
                     u_ref: np.ndarray | None = None,
                     B_ref=None, C_ref=None,
                     workspace: "hlo.HelmWorkspace | None" = None,
                     n_workers: int | None = None,
                     verbose: bool = False):
    """Solve the Helmholtz impedance ExpMsFEM problem.

    Returns a dict with {"u_ms_fine", "u_ref_fine", "e_L2", "e_H1",
    "e_H1_matlab", "B_ref", "C_ref"}.
    """
    count = 4 + 4 * (N_e + 1)
    n_cells = N_c * N_c
    n_fine_nodes = (N_f + 1) ** 2

    if workspace is None:
        ws = hlo.HelmWorkspace(N_c, N_f, k0)
        t0 = time.time()
        ws.prefactor_all(n_workers=n_workers)
        if verbose:
            print(f"  prefactor LUs: {time.time() - t0:.2f}s "
                  f"({len(ws._cell)} cells, {len(ws._patch)} patches)")
    else:
        ws = workspace
        ws._edge_cache = {}

    t0 = time.time()
    heb.prefactor_edges(ws, N_e, n_workers=n_workers)
    if verbose:
        print(f"  prefactor edges: {time.time() - t0:.2f}s "
              f"({len(ws._edge_cache)} edges)")

    values = np.empty((n_cells, n_fine_nodes, count), dtype=np.complex128)
    K_elems = np.empty((n_cells, count, count), dtype=np.complex128)
    f_elems = np.empty((n_cells, count), dtype=np.complex128)
    bubbles = np.empty((n_cells, n_fine_nodes), dtype=np.complex128)

    def _one_cell(job):
        cell, m, n = job
        value, _, K_elem, f_elem = heb.element_basis(ws, m, n, N_e)
        bub = hlo.bubble(m, n, N_c, N_f, k0)
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

    cell_idx = all_cell_global_indices(N_c, N_e)
    N_total = n_total(N_c, N_e)
    A_glob, F_glob = hass.assemble_global(K_elems, f_elems, cell_idx, N_total)
    bdy = hass.domain_boundary_dofs_helm(N_c, N_e)
    u_global = hass.solve_coarse(A_glob, F_glob, bdy)

    u_ms_fine = reconstruct_fine(values, u_global, bubbles, N_c, N_f, N_e)

    # Reference
    if u_ref is None or B_ref is None or C_ref is None:
        u_ref, B_ref, C_ref, _ = hfem.solve_fine(N_c * N_f, k0)

    # Errors (Matlab err: sesquilinear)
    e = u_ref - u_ms_fine
    num_L2 = float(np.real(e.conj() @ (C_ref @ e)))
    den_L2 = float(np.real(u_ref.conj() @ (C_ref @ u_ref)))
    e_L2 = np.sqrt(num_L2 / den_L2)
    num_H1 = float(np.real(e.conj() @ (B_ref @ e)))
    den_H1 = float(np.real(u_ref.conj() @ (B_ref @ u_ref)))
    e_H1 = np.sqrt(num_H1 / den_H1) if den_H1 > 0 else np.inf
    # Matlab form: sqrt(e'*K*e / sqrt(u'*K*u))  (odd but reproduced)
    e_H1_m = np.sqrt(num_H1 / np.sqrt(den_H1)) if den_H1 > 0 else np.inf

    return {
        "u_ms_fine": u_ms_fine,
        "u_ref_fine": u_ref,
        "e_L2": e_L2,
        "e_H1": e_H1,
        "e_H1_matlab": e_H1_m,
        "B_ref": B_ref,
        "C_ref": C_ref,
    }
