"""Fine-scale FEM reference solver — convergence & sanity tests."""

import numpy as np

from expmsfem import fem
from expmsfem.coefficients import afun_constant


def test_constant_coef_sanity():
    """With a ≡ 1 and RHS ≡ -1 on [0,1]^2, Dirichlet-0, the Q1 solution at
    the centre should be close to the true analytical value. The true solution
    of -Δu = -1, u=0 on the unit square is u(x,y) = sum of series; at the
    centre u(0.5, 0.5) ≈ -0.0736713... (classic Poisson square)."""
    u, A, M, F = fem.solve_fine(afun_constant, N_f=64)
    u2d = u.reshape(65, 65)
    center_val = u2d[32, 32]
    # Known Fourier solution for -Δu = 1 on [0,1]^2, u=0 on boundary, at centre:
    # u(0.5, 0.5) = (16/π^4) Σ_{odd m,n} sin(mπ/2) sin(nπ/2) / (m n (m^2+n^2))
    # Evaluating to high precision gives ≈ 0.0736713520...
    # Our rhs is -1, so expect center_val ≈ -0.0736713520
    expected = -0.07367135328
    assert abs(center_val - expected) < 5e-4


def test_symmetry():
    u, A, M, F = fem.solve_fine(afun_constant, N_f=16)
    u2d = u.reshape(17, 17)
    np.testing.assert_allclose(u2d, u2d.T, atol=1e-12)
    np.testing.assert_allclose(u2d, u2d[::-1, :], atol=1e-12)
    np.testing.assert_allclose(u2d, u2d[:, ::-1], atol=1e-12)


def test_boundary_zero():
    u, _, _, _ = fem.solve_fine(afun_constant, N_f=8)
    u2d = u.reshape(9, 9)
    np.testing.assert_array_equal(u2d[0, :], 0)
    np.testing.assert_array_equal(u2d[-1, :], 0)
    np.testing.assert_array_equal(u2d[:, 0], 0)
    np.testing.assert_array_equal(u2d[:, -1], 0)


def test_energy_norm_finite():
    """u'·A·u should be positive and u'·M·u should be positive."""
    u, A, M, _ = fem.solve_fine(afun_constant, N_f=16)
    assert u @ (A @ u) > 0
    assert u @ (M @ u) > 0


def test_convergence_order():
    """Approximate O(h^2) convergence of Q1 FEM on smooth problem."""
    errs = []
    for N_f in (16, 32, 64):
        u, _, M, _ = fem.solve_fine(afun_constant, N_f=N_f)
        # True solution on [0,1]^2 for -Δu = 1 (note: we solve -Δu = -1, sign
        # just flips). We compare via centre-value convergence, which goes
        # like O(h^2). Use |u(0.5,0.5) - u_true(0.5,0.5)| as error proxy.
        u2d = u.reshape(N_f + 1, N_f + 1)
        e = abs(u2d[N_f // 2, N_f // 2] - (-0.07367135328))
        errs.append(e)
    # Each doubling of N should reduce error by ~4×
    r1 = errs[0] / errs[1]
    r2 = errs[1] / errs[2]
    assert 2.5 < r1 < 6, f"expected ~4x rate, got {r1:.2f}"
    assert 2.5 < r2 < 6, f"expected ~4x rate, got {r2:.2f}"
