"""End-to-end Schrödinger ExpMsFEM driver.

Builds the coarse Hamiltonian `H_c` and mass `M_c` from per-cell ExpMsFEM
bases and solves the dense generalised eigenproblem `H_c c = E M_c c`.
The DOF layout extends the elliptic one: shared nodal + shared edge DOFs
as before, **plus** `K_int` cell-unique interior-eigenmode DOFs per cell
appended at the end of the global vector.

Dirichlet-zero boundary condition on `∂Ω` is enforced by zeroing out
rows/columns of the boundary-side DOFs (nodal corners + horizontal
edges at `k ∈ {0, N_c}` + vertical edges at `m ∈ {0, N_c}`). Cell-interior
DOFs never sit on `∂Ω`, so they're kept.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from typing import Callable

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from ..assembly import (
    cell_global_indices, domain_boundary_dofs, n_nodal, n_total,
)
from . import element_basis as heb
from . import fem as hfem
from . import local_ops as hlo


def _global_dof_layout(N_c: int, N_e: int, K_int: int):
    """Return
        n_shared : # shared DOFs (nodal + edges, uses elliptic layout with
                   n_per_edge = N_e)
        n_total : n_shared + N_c² · K_int
        cell_indices : (N_c², count) — local-slot → global-index map
                       where count = 4 + 4·N_e + K_int and the last
                       K_int slots point to this cell's unique interior DOFs
        dirichlet : DOFs to pin to zero at the coarse level
    """
    n_shared = n_total(N_c, N_e, n_per_edge=N_e)
    N_total = n_shared + (N_c * N_c) * K_int
    count = 4 + 4 * N_e + K_int
    indices = np.empty((N_c * N_c, count), dtype=np.int64)
    for n in range(N_c):
        for m in range(N_c):
            cell = n * N_c + m
            # Shared (elliptic-layout) portion: first 4 + 4*N_e slots
            shared = cell_global_indices(m, n, N_c, N_e, n_per_edge=N_e)
            indices[cell, : 4 + 4 * N_e] = shared
            # Unique interior DOFs: append at the tail
            base = n_shared + cell * K_int
            indices[cell, 4 + 4 * N_e :] = base + np.arange(K_int)
    bdy = domain_boundary_dofs(N_c, N_e, n_per_edge=N_e)
    return n_shared, N_total, indices, bdy


def run_schrodinger_expmsfem(V_fun: Callable, N_c: int, N_f: int, N_e: int,
                             k: int = 6, K_int: int = 3,
                             E_ref=None, psi_ref=None,
                             H_ref=None, M_ref=None,
                             workspace: "hlo.Workspace | None" = None,
                             n_workers: int | None = None,
                             verbose: bool = False):
    """Lowest `k` eigenpairs of `H ψ = E ψ` via ExpMsFEM on `[0, 1]²`.

    Parameters
    ----------
    V_fun : (x, y) ↦ V(x, y).
    N_c, N_f : coarse and fine grid sizes.
    N_e : per-edge eigen-modes.
    K_int : per-cell internal Dirichlet eigenfunctions (default 3).
    k : number of coarse eigenpairs to return.
    E_ref, psi_ref, H_ref, M_ref : optional fine-scale reference (skip
        the internal `solve_fine_eigenpairs` call).

    Returns dict with keys
        "E", "psi_ms_fine", "E_ref", "psi_ref_fine", "H_ref", "M_ref".
    `psi_ms_fine` has shape `((N_c·N_f + 1)², k)` with Dirichlet-0 BC.
    """
    count = 4 + 4 * N_e + K_int
    n_cells = N_c * N_c
    n_fine = (N_f + 1) ** 2

    # --- Offline ---
    if workspace is None:
        ws = hlo.Workspace(V_fun, N_c, N_f)
        t0 = time.time()
        ws.prefactor_all(n_workers=n_workers)
        if verbose:
            print(f"  prefactor LUs: {time.time() - t0:.2f}s "
                  f"({len(ws._cell)} cells, {len(ws._patch)} patches)")
    else:
        ws = workspace

    t0 = time.time()
    heb.prefactor_edges(ws, N_e, n_workers=n_workers)
    if verbose:
        print(f"  prefactor edges: {time.time() - t0:.2f}s "
              f"({len(ws._edge_cache)} edges)")

    values = np.empty((n_cells, n_fine, count), dtype=np.float64)
    H_elems = np.empty((n_cells, count, count), dtype=np.float64)
    M_elems = np.empty((n_cells, count, count), dtype=np.float64)

    def _one_cell(job):
        cell, m, n = job
        value, H_elem, M_elem = heb.element_basis(ws, m, n, N_e, K_int)
        return cell, value, H_elem, M_elem

    jobs = [(n * N_c + m, m, n) for n in range(N_c) for m in range(N_c)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for cell, value, H_elem, M_elem in pool.map(_one_cell, jobs):
            values[cell] = value
            H_elems[cell] = H_elem
            M_elems[cell] = M_elem
    if verbose:
        print(f"  per-cell bases: {time.time() - t0:.2f}s")

    # --- Global assembly + Dirichlet ---
    n_shared, N_total, cell_idx, bdy = _global_dof_layout(N_c, N_e, K_int)
    I = np.broadcast_to(cell_idx[:, :, None], (n_cells, count, count)).reshape(-1)
    J = np.broadcast_to(cell_idx[:, None, :], (n_cells, count, count)).reshape(-1)
    H_coarse = sp.coo_matrix((H_elems.reshape(-1), (I, J)),
                             shape=(N_total, N_total)).tocsc()
    M_coarse = sp.coo_matrix((M_elems.reshape(-1), (I, J)),
                             shape=(N_total, N_total)).tocsc()

    interior = np.setdiff1d(np.arange(N_total), bdy, assume_unique=True)
    H_int = H_coarse[interior][:, interior].toarray()
    M_int = M_coarse[interior][:, interior].toarray()
    E, C = sla.eigh(0.5 * (H_int + H_int.T), 0.5 * (M_int + M_int.T))
    E = E[:k]
    C = C[:, :k]

    psi_coarse_full = np.zeros((N_total, k), dtype=np.float64)
    psi_coarse_full[interior, :] = C

    # --- Reconstruction ---
    N_fine_side = N_c * N_f
    psi_ms = np.zeros((N_fine_side + 1, N_fine_side + 1, k), dtype=np.float64)
    for n in range(N_c):
        for m in range(N_c):
            cell = n * N_c + m
            gidx = cell_idx[cell]
            coefs = psi_coarse_full[gidx, :]
            zs = values[cell] @ coefs
            zs2 = zs.reshape(N_f + 1, N_f + 1, k)
            psi_ms[n * N_f : n * N_f + N_f + 1,
                   m * N_f : m * N_f + N_f + 1, :] = zs2
    psi_ms_fine = psi_ms.reshape((N_fine_side + 1) ** 2, k)

    # --- Reference ---
    if E_ref is None or H_ref is None or M_ref is None:
        E_ref, psi_ref, H_ref, M_ref = hfem.solve_fine_eigenpairs(
            V_fun, N_fine_side, k=max(k, 6),
        )

    return {
        "E": E,
        "psi_ms_fine": psi_ms_fine,
        "E_ref": E_ref,
        "psi_ref_fine": psi_ref,
        "H_coarse": H_coarse,
        "M_coarse": M_coarse,
        "H_ref": H_ref,
        "M_ref": M_ref,
    }
