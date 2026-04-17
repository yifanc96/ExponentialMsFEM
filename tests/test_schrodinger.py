"""Tests for the Schrödinger ExpMsFEM submodule.

Sanity checks against the 2D quantum harmonic oscillator, whose exact
eigenvalues `E_{n_x, n_y} = (n_x + n_y + 1) ω` are well-approximated by
the box-confined problem once `ω` is large enough for the ground-state
Gaussian to fit well inside `[0, 1]²`.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pytest

from expmsfem.schrodinger import local_ops as hlo
from expmsfem.schrodinger.driver import run_schrodinger_expmsfem
from expmsfem.schrodinger.fem import solve_fine_eigenpairs
from expmsfem.schrodinger.potentials import V_harmonic_oscillator


def test_fine_fem_harmonic_oscillator_eigenvalues():
    """Fine-scale Q1 FEM recovers the analytic E_{n_x, n_y} = (n_x+n_y+1)ω
    eigenvalues of the 2D harmonic oscillator to within Q1 discretisation
    error (~0.1% at N_f=128, ω=50)."""
    omega = 50.0
    V = lambda x, y: V_harmonic_oscillator(x, y, omega=omega)
    E, psi, H, M = solve_fine_eigenpairs(V, N_f=128, k=6, sigma=0.0)
    # Analytic (all of R²): E_0 = ω, E_1 = E_2 = 2ω, E_3 = E_4 = E_5 = 3ω
    analytic = np.array([1 * omega, 2 * omega, 2 * omega,
                         3 * omega, 3 * omega, 3 * omega])
    rel_err = np.abs(E - analytic) / analytic
    assert rel_err.max() < 5e-3, f"E = {E}, rel err = {rel_err}"
    # M-orthogonality of returned eigenvectors
    G = psi.T @ (M @ psi)
    np.testing.assert_allclose(G, np.eye(6), atol=1e-8)


def test_expmsfem_harmonic_oscillator_converges_in_K_int():
    """At fixed N_e=3, increasing K_int drives the ground-state error down
    on the harmonic oscillator."""
    omega = 50.0
    V = lambda x, y: V_harmonic_oscillator(x, y, omega=omega)
    E_ref, psi_ref, H_ref, M_ref = solve_fine_eigenpairs(V, 128, k=6, sigma=0.0)
    ws = hlo.Workspace(V, N_c=8, N_f=16)
    ws.prefactor_all(n_workers=2)

    errs = []
    for K_int in [1, 3, 5, 8]:
        out = run_schrodinger_expmsfem(
            V, 8, 16, N_e=3, k=6, K_int=K_int,
            E_ref=E_ref, H_ref=H_ref, M_ref=M_ref,
            workspace=ws, n_workers=2,
        )
        errs.append(abs(out["E"][0] - E_ref[0]) / E_ref[0])
    errs = np.array(errs)
    # Monotone decrease (allow tiny wobble via <= 1e-10)
    assert np.all(np.diff(errs) < 1e-10), f"errs = {errs}"
    # Strong improvement: K_int=1 vs K_int=8 differ by ≥ 5x
    assert errs[0] / errs[-1] > 5, f"errs = {errs}"
    # Final accuracy better than 1e-3
    assert errs[-1] < 1e-3


def test_expmsfem_harmonic_oscillator_smoke_tiny():
    """Fast smoke test: moderate accuracy on a 4×4 coarse grid."""
    omega = 30.0
    V = lambda x, y: V_harmonic_oscillator(x, y, omega=omega)
    E_ref, psi_ref, H_ref, M_ref = solve_fine_eigenpairs(V, 64, k=4, sigma=0.0)
    out = run_schrodinger_expmsfem(
        V, 4, 16, N_e=3, k=4, K_int=5,
        E_ref=E_ref, H_ref=H_ref, M_ref=M_ref,
        n_workers=2,
    )
    # All four lowest eigenvalues within 5% of the fine reference
    rel = np.abs(out["E"] - E_ref) / E_ref
    assert rel.max() < 5e-2, f"rel = {rel}"


def test_element_basis_shapes():
    """Per-cell basis array has the expected shape (4 nodal + 4·N_e edge
    + K_int cell-interior columns)."""
    from expmsfem.schrodinger.element_basis import element_basis, prefactor_edges
    omega = 30.0
    V = lambda x, y: V_harmonic_oscillator(x, y, omega=omega)
    N_c, N_f, N_e, K_int = 4, 8, 2, 3
    ws = hlo.Workspace(V, N_c, N_f)
    ws.prefactor_all(n_workers=2)
    prefactor_edges(ws, N_e, n_workers=2)
    value, H_elem, M_elem = element_basis(ws, 1, 1, N_e, K_int)
    count = 4 + 4 * N_e + K_int
    assert value.shape == ((N_f + 1) ** 2, count)
    assert H_elem.shape == (count, count)
    assert M_elem.shape == (count, count)
    # Symmetry of H and M
    np.testing.assert_allclose(H_elem, H_elem.T, atol=1e-10)
    np.testing.assert_allclose(M_elem, M_elem.T, atol=1e-10)
