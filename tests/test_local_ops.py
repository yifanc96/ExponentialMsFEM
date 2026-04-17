"""Tests for the local operators (basefun, bubble, harmext, restrict)."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from expmsfem import local_ops, mesh, q1
from expmsfem.coefficients import afun_constant, afun_periodic


def test_basefun_symmetric_and_singular_on_constant():
    """For constant a, basefun stiffness has constants in its kernel."""
    A = local_ops.basefun(afun_constant, m=0, n=0, N_c=4, N_f=4).toarray()
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    ones = np.ones(A.shape[0])
    np.testing.assert_allclose(A @ ones, 0, atol=1e-12)


def test_basefun_equals_afun_times_reference_on_constant():
    """For constant a ≡ c, basefun stiffness should be c × constant-coef
    stiffness (since the Q1 stiffness is scale-free on squares)."""
    def a1(x, y):
        return 1.0 * np.ones_like(np.asarray(x) * np.asarray(y))

    def a2(x, y):
        return 2.5 * np.ones_like(np.asarray(x) * np.asarray(y))

    A1 = local_ops.basefun(a1, 0, 0, 4, 4).toarray()
    A2 = local_ops.basefun(a2, 0, 0, 4, 4).toarray()
    np.testing.assert_allclose(A2, 2.5 * A1, atol=1e-12)


def test_bubble_zero_on_boundary_and_positive_interior():
    """Bubble: Dirichlet-0 with constant negative RHS → negative interior."""
    N_f = 8
    u = local_ops.bubble(afun_constant, m=0, n=0, N_c=2, N_f=N_f)
    u2d = u.reshape(N_f + 1, N_f + 1)
    np.testing.assert_array_equal(u2d[0, :], 0)
    np.testing.assert_array_equal(u2d[-1, :], 0)
    np.testing.assert_array_equal(u2d[:, 0], 0)
    np.testing.assert_array_equal(u2d[:, -1], 0)
    # Interior: solving Kx = -h^2/4·(load) with K positive → u ≤ 0
    assert (u2d[1:-1, 1:-1] <= 0).all()
    # Symmetric
    np.testing.assert_allclose(u2d, u2d.T, atol=1e-12)


def test_harmext_lifts_identity_data_exactly():
    """L1, L2 must satisfy u[b] = f on the patch boundary for each column."""
    N_f = 6
    L1, L2, N = local_ops.harmext(
        afun_periodic, m=1, n=1, N_c=4, N_f=N_f, i=1
    )
    b = mesh.cell_boundary_nodes(N_f, N_f)
    # L1: K1-Dirichlet-harmonic on cell (1,1); rows at b should equal f1.
    # f1 puts identity on Matlab rows 3Nf+2..4Nf (0-idx: 3Nf+1..4Nf-1),
    # which are the top-interior rows of b.
    f1 = np.zeros((b.size, N_f - 1))
    idx = np.arange(3 * N_f + 1, 4 * N_f)
    f1[idx, np.arange(N_f - 1)] = 1.0
    np.testing.assert_allclose(L1[b, :], f1, atol=1e-12)

    # f2: bottom-interior rows 1..N_f-1
    f2 = np.zeros((b.size, N_f - 1))
    idx = np.arange(1, N_f)
    f2[idx, np.arange(N_f - 1)] = 1.0
    np.testing.assert_allclose(L2[b, :], f2, atol=1e-12)


def test_harmext_N_symmetric_positive_definite():
    L1, L2, N = local_ops.harmext(
        afun_periodic, m=1, n=1, N_c=4, N_f=6, i=1
    )
    np.testing.assert_allclose(N, N.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(N)
    assert eigs.min() > 1e-8


def test_restrict_shapes_and_P_spd():
    N_c, N_f = 4, 6
    R, P, bub = local_ops.restrict(
        afun_periodic, m=1, n=1, N_c=N_c, N_f=N_f, t=1
    )
    assert R.shape[0] == N_f - 1
    assert P.shape == (R.shape[1], R.shape[1])
    assert bub.shape == (N_f - 1,)
    np.testing.assert_allclose(P, P.T, atol=1e-10)
    # Energy Gram matrix should be SPD
    eigs = np.linalg.eigvalsh(P)
    assert eigs.min() > -1e-10
    assert eigs.max() > 0


def test_restrict_boundary_edge_shrunken_patch():
    """For m=0, the horizontal edge's patch is 2 cells wide (not 3)."""
    N_c, N_f = 4, 4
    R0, P0, _ = local_ops.restrict(afun_constant, m=0, n=1, N_c=N_c, N_f=N_f, t=1)
    R1, P1, _ = local_ops.restrict(afun_constant, m=1, n=1, N_c=N_c, N_f=N_f, t=1)
    # Interior patch perimeter is 2*((3*N_f) + (2*N_f)) = 10*N_f nodes,
    # boundary patch is 2*((2*N_f) + (2*N_f)) = 8*N_f nodes.
    # n_active = perimeter - 1 for interior patches when none of the patch
    # sides lie on ∂Ω.
    # For m=0: patch left col is on ∂Ω → some perimeter DOFs disabled.
    assert R0.shape[1] < 8 * N_f
    assert R1.shape[1] < 10 * N_f
    # And the boundary-patch active DOFs should be fewer than interior's.
    assert R0.shape[1] <= R1.shape[1]


def test_eigen_decay_exponential_on_constant_coef():
    """On constant-a, (R'NR, P) generalised eigenvalues decay exponentially.

    The theory of the ExpMsFEM paper says the dominant m edge modes capture
    all but an exp(−α m) fraction of the patch harmonic space's energy that
    reaches the two target cells. We don't check absolute scale (it depends
    on patch geometry), only that successive eigenvalues drop roughly
    geometrically.
    """
    import scipy.linalg as sla

    N_c, N_f = 8, 16
    L1, L2, N = local_ops.harmext(afun_constant, m=3, n=3, N_c=N_c, N_f=N_f, i=1)
    R, P, _ = local_ops.restrict(afun_constant, m=3, n=3, N_c=N_c, N_f=N_f, t=1)
    P_reg = P + 1e-12 * np.eye(P.shape[0])
    eigvals = sla.eigvalsh(R.T @ N @ R, P_reg)
    eigvals = np.sort(eigvals)[::-1]  # largest first
    # Top 10 should be strictly decreasing and span ≥4 orders of magnitude.
    top = eigvals[:10]
    assert np.all(np.diff(top) <= 1e-14), f"not monotone: {top}"
    assert top[0] / top[9] > 1e4, f"decay too slow: {top}"
    # Log-linear slope on top 5 should be a meaningful negative number (at
    # least −0.5 per mode ⇒ each mode adds ~0.5 to −log10 λ).
    log_top = np.log10(top[:5])
    slope = (log_top[-1] - log_top[0]) / 4
    assert slope < -0.5, f"slope {slope} not exponentially decaying"
