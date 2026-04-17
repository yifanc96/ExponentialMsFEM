"""Tests for mesh / q1 / coefficients primitives."""

import numpy as np

from expmsfem import mesh, q1, coefficients


def _matlab_loc2glo(N, m, n, i):
    """1-indexed Matlab loc2glo(N,m,n,i), for cross-checking."""
    if i <= 2:
        return (N + 1) * (n - 1) + m + i - 1
    else:
        return (N + 1) * n + m + 4 - i


def test_node_index_matches_matlab():
    N = 5
    for m in range(1, N + 1):
        for n in range(1, N + 1):
            for i in range(1, 5):
                expected_1idx = _matlab_loc2glo(N, m, n, i)
                if i == 1:
                    ours = mesh.node_index(N, m - 1, n - 1)
                elif i == 2:
                    ours = mesh.node_index(N, m, n - 1)
                elif i == 3:
                    ours = mesh.node_index(N, m, n)
                else:
                    ours = mesh.node_index(N, m - 1, n)
                assert ours == expected_1idx - 1


def test_local_to_global_nodes_matches_matlab_per_element():
    N = 4
    g = mesh.local_to_global_nodes(N, N)
    # Check a few cells
    for m in range(1, N + 1):
        for n in range(1, N + 1):
            cell_idx_row_major = (n - 1) * N + (m - 1)
            expected = np.array(
                [_matlab_loc2glo(N, m, n, i) - 1 for i in range(1, 5)]
            )
            np.testing.assert_array_equal(g[cell_idx_row_major], expected)


def test_boundary_nodes_count_and_set():
    Nx, Ny = 4, 3
    b = mesh.cell_boundary_nodes(Nx, Ny)
    # All perimeter nodes; total perimeter = 2*(Nx+Ny)
    assert b.size == 2 * (Nx + Ny)
    n = (Nx + 1) * (Ny + 1)
    assert np.all(b >= 0) and np.all(b < n)
    assert np.unique(b).size == b.size
    # Every perimeter node (i,j) where i in {0,Nx} or j in {0,Ny} is listed
    ii, jj = np.meshgrid(np.arange(Nx + 1), np.arange(Ny + 1), indexing="xy")
    perim = (ii == 0) | (ii == Nx) | (jj == 0) | (jj == Ny)
    expected_set = set((jj * (Nx + 1) + ii)[perim].tolist())
    assert set(b.tolist()) == expected_set


def test_boundary_nodes_matches_matlab_ordering():
    """Order: [bottom row Nx+1 entries, left-col interior Ny,
    right-col interior Ny, top-row interior Nx-1]."""
    Nx, Ny = 5, 5
    b = mesh.cell_boundary_nodes(Nx, Ny)
    # Matlab (1-indexed):
    # [1:Nx+1, Nx+2:Nx+1:(Nx+1)*(Ny+1),
    #  2*(Nx+1):Nx+1:(Nx+1)*(Ny+1),
    #  Nx*Ny+Ny+2:(Nx+1)*(Ny+1)-1]
    expected = np.concatenate(
        [
            np.arange(1, Nx + 2),
            np.arange(Nx + 2, (Nx + 1) * (Ny + 1) + 1, Nx + 1),
            np.arange(2 * (Nx + 1), (Nx + 1) * (Ny + 1) + 1, Nx + 1),
            np.arange(Nx * Ny + Ny + 2, (Nx + 1) * (Ny + 1)),
        ]
    )
    np.testing.assert_array_equal(b, expected - 1)


def test_q1_reference_matrices_symmetric():
    np.testing.assert_allclose(q1.C_STIFF, q1.C_STIFF.T)
    np.testing.assert_allclose(q1.C_MASS, q1.C_MASS.T)
    # Stiffness with a=1 has null space = constants
    K = q1.C_STIFF
    null = np.ones(4)
    np.testing.assert_allclose(K @ null, 0, atol=1e-14)
    # Row sums of mass (on unit square) = 1/4 (since ∫φ_i = h^2/4 with h=1)
    np.testing.assert_allclose(q1.C_MASS.sum(axis=1), 0.25)


def test_afun_periodic_matches_matlab_at_known_point():
    # Spot check: Matlab evaluates afun(0.25, 0.75) and afun(0.5, 0.5)
    # Reference values computed by hand-substitution below.
    def matlab_afun(t, s):
        eps = [1 / 5, 1 / 13, 1 / 17, 1 / 31, 1 / 65]
        a = (
            (1.1 + np.sin(2 * np.pi * t / eps[0]))
            / (1.1 + np.sin(2 * np.pi * s / eps[0]))
            + (1.1 + np.sin(2 * np.pi * s / eps[1]))
            / (1.1 + np.cos(2 * np.pi * t / eps[1]))
            + (1.1 + np.cos(2 * np.pi * t / eps[2]))
            / (1.1 + np.sin(2 * np.pi * s / eps[2]))
            + (1.1 + np.sin(2 * np.pi * s / eps[3]))
            / (1.1 + np.cos(2 * np.pi * t / eps[3]))
            + (1.1 + np.cos(2 * np.pi * t / eps[4]))
            / (1.1 + np.sin(2 * np.pi * s / eps[4]))
            + np.sin(4 * s**2 * t**2)
            + 1.0
        )
        return a / 6.0

    for t, s in [(0.25, 0.75), (0.5, 0.5), (0.1, 0.9), (0.333, 0.666)]:
        assert np.isclose(coefficients.afun_periodic(t, s), matlab_afun(t, s))

    # Vectorised call
    tv = np.array([0.25, 0.5, 0.1])
    sv = np.array([0.75, 0.5, 0.9])
    av = coefficients.afun_periodic(tv, sv)
    for i in range(3):
        assert np.isclose(av[i], matlab_afun(tv[i], sv[i]))

    # Positivity (should always be > 0 on [0,1]^2 for this formula)
    grid = np.linspace(0.01, 0.99, 25)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    A = coefficients.afun_periodic(xx, yy)
    assert (A > 0).all()


def test_random_field_matches_matlab_formula_at_grid_nodes():
    """At integer multiples of 1/M, bilinear interpolation should return the
    underlying grid value exactly (with alpha_t = 0)."""
    M = 8
    goo = coefficients.generate_random_field(M=M, seed=42)
    a = coefficients.afun_random(goo, M=M)
    for i in range(M):
        for j in range(M):
            t = i / M
            s = j / M
            # goo[ti, si] with ti = floor(M*t) = i, si = j
            expected = goo[i, j]
            assert np.isclose(a(t, s), expected)


def test_random_field_bilinear_interior_point():
    """Interpolate at the centre of one cell and cross-check against hand
    bilinear formula."""
    M = 8
    goo = coefficients.generate_random_field(M=M, seed=42)
    a = coefficients.afun_random(goo, M=M)
    # Centre of grid cell (2, 3): t = (2 + 0.5)/M, s = (3 + 0.5)/M
    t = 2.5 / M
    s = 3.5 / M
    alpha_t = 0.5
    alpha_s = 0.5
    expected = (
        (1 - alpha_s) * (1 - alpha_t) * goo[2, 3]
        + (1 - alpha_s) * alpha_t * goo[3, 3]
        + alpha_s * (1 - alpha_t) * goo[2, 4]
        + alpha_s * alpha_t * goo[3, 4]
    )
    assert np.isclose(a(t, s), expected)


def test_random_field_vectorised():
    M = 16
    goo = coefficients.generate_random_field(M=M, seed=1)
    a = coefficients.afun_random(goo, M=M)
    tv = np.linspace(0.05, 0.95, 7)
    sv = np.linspace(0.1, 0.9, 7)
    T, S = np.meshgrid(tv, sv, indexing="xy")
    A = a(T, S)
    # Same result by individual evaluation
    A_ref = np.array([[a(t, s) for t in tv] for s in sv])
    np.testing.assert_allclose(A, A_ref)
    # Positivity
    assert (A > 0).all()


def test_highcontrast_background_and_inclusions():
    # Centre (0.2, 0.2) is an inclusion centre → a = 64
    assert coefficients.afun_highcontrast(0.2, 0.2) == 64
    # Clearly outside any inclusion
    assert coefficients.afun_highcontrast(0.05, 0.05) == 1
    # Slightly off-centre but within radius: distance sqrt(0.01^2+0.01^2) ≈ 0.014 < 0.015
    assert coefficients.afun_highcontrast(0.21, 0.21) == 64
    # Outside radius: distance 0.02 > 0.015
    assert coefficients.afun_highcontrast(0.22, 0.22) == 1


def test_highcontrast_vectorised():
    grid = np.linspace(0.01, 0.99, 50)
    X, Y = np.meshgrid(grid, grid, indexing="xy")
    A = coefficients.afun_highcontrast(X, Y)
    assert A.shape == X.shape
    # Only two distinct values
    uniq = np.unique(A)
    assert set(uniq.tolist()) <= {1.0, 64.0}
    # Most of the domain is background; inclusions are small disks
    # 49 disks of radius 0.015 → analytic area ≈ 49π·0.015² ≈ 0.035.
    # Use a fine grid so the sampling-error band is small.
    fine = np.linspace(0.005, 0.995, 400)
    X2, Y2 = np.meshgrid(fine, fine, indexing="xy")
    A2 = coefficients.afun_highcontrast(X2, Y2)
    frac_bg = (A2 == 1.0).mean()
    assert 0.96 < frac_bg < 0.985
