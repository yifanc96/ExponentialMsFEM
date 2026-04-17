"""Tests for Helmholtz local operators."""

import numpy as np

from expmsfem.helmholtz import local_ops as hlo
from expmsfem.helmholtz import fem as hfem


def test_dirichlet_indices_interior_cell():
    """Interior cell — all 4 sides are interior-to-domain → Dirichlet = full
    perimeter = 4*N_f DOFs."""
    N_f = 8
    bdy = hlo.dirichlet_indices(N_c_x=4, N_c_y=4, N_x=N_f, N_y=N_f, m=2, n=2)
    assert bdy.size == 4 * N_f
    assert bdy.min() >= 0 and bdy.max() < (N_f + 1) ** 2


def test_dirichlet_indices_boundary_cell_left():
    """Left-boundary cell keeps its LEFT side free (impedance); remaining 3
    sides are Dirichlet. Size < 4*N_f."""
    N_f = 8
    bdy_interior = hlo.dirichlet_indices(4, 4, N_f, N_f, 2, 2)
    bdy_left = hlo.dirichlet_indices(4, 4, N_f, N_f, 1, 2)
    assert bdy_left.size < bdy_interior.size
    assert bdy_left.size > 0


def test_basefun_symmetric_complex():
    """Cell stiffness should be complex-symmetric (K = K.T)."""
    A = hlo.basefun(m=1, n=1, N_c=4, N_f=4, k0=2.0)
    dense = A.toarray()
    np.testing.assert_allclose(dense, dense.T, atol=1e-12)


def test_workspace_caches_factors():
    ws = hlo.HelmWorkspace(N_c=4, N_f=4, k0=2.0)
    ws.prefactor_all(n_workers=2)
    # Verify all cells and interior edges have factors built.
    assert len(ws._cell) == 16
    assert len(ws._patch) == 4 * 3 + 4 * 3  # (N_c-1)*N_c + N_c*(N_c-1) patches
    # Random cell factor access doesn't crash
    A, bdy, fac, fac_T = ws.cell_factor(2, 2)
    assert A.shape[0] == (4 + 1) ** 2


def test_bubble_runs_and_zero_on_dirichlet_set():
    """Bubble solution should exactly match bdy_values (zero) on Dirichlet
    nodes. For Helmholtz ffun=0 so interior bubble is purely from boundary
    impedance data — zero for interior cells, non-trivial for boundary cells.
    """
    ws = hlo.HelmWorkspace(N_c=4, N_f=4, k0=2.0)
    u_interior = hlo.bubble(m=2, n=2, N_c=4, N_f=4, k0=2.0)
    # Interior cell: Dirichlet = full perimeter, load = 0 (ffun=0, gfun=0
    # since cell doesn't touch ∂Ω), so solution ≡ 0.
    assert np.max(np.abs(u_interior)) < 1e-12

    # Boundary cell (m=0 which is the left column): gfun on left edge is
    # non-zero → bubble is non-trivial.
    u_bdy = hlo.bubble(m=0, n=2, N_c=4, N_f=4, k0=2.0)
    assert np.max(np.abs(u_bdy)) > 1e-4


def test_harmext_shapes_and_symmetry_of_N():
    """harmext returns (L1, L2, N). N should be Hermitian and PSD."""
    ws = hlo.HelmWorkspace(N_c=4, N_f=4, k0=2.0)
    ws.prefactor_all(n_workers=2)
    L1, L2, N = hlo.harmext(ws, m=1, n=1, t=1)
    assert L1.shape == (5 * 5, 4 - 1)
    assert L2.shape == L1.shape
    assert N.shape == (3, 3)
    # Complex-Hermitian
    np.testing.assert_allclose(N, N.conj().T, atol=1e-10)


def test_restrict_shapes():
    ws = hlo.HelmWorkspace(N_c=6, N_f=4, k0=2.0)
    ws.prefactor_all(n_workers=2)
    R, P, bub = hlo.restrict(ws, m=2, n=2, t=1)
    assert R.shape[0] == 4 - 1  # N_f - 1
    assert P.shape == (R.shape[1], R.shape[1])
    assert bub.shape == (4 - 1,)


def test_fine_fem_helmholtz_basic():
    """Smoke test: fine Helmholtz FEM runs and produces a sensible solution
    with a plane-wave boundary load."""
    u, B, C, F = hfem.solve_fine(N_f=16, k0=2.0)
    assert u.dtype == np.complex128
    # u should have non-trivial complex part (plane wave)
    assert np.abs(u.imag).max() > 1e-3
    # Bounded
    assert np.abs(u).max() < 20
