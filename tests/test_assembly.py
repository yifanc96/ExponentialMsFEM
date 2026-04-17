"""Tests for global DOF layout and assembly."""

import numpy as np

from expmsfem import assembly


def test_total_dof_count_matches_matlab_formula():
    for N_c, N_e in [(4, 2), (8, 3), (32, 7)]:
        N_p = N_e + 1
        expected = (N_c + 1) ** 2 + 2 * N_c * (N_c + 1) * N_p
        assert assembly.n_total(N_c, N_e) == expected


def test_cell_indices_nodal_corners_match_node_indexing():
    """Corners of cell (m,n) in counterclockwise order must be the 4 node
    indices (m,n), (m+1,n), (m+1,n+1), (m,n+1)."""
    N_c, N_e = 4, 2
    for m in range(N_c):
        for n in range(N_c):
            idx = assembly.cell_global_indices(m, n, N_c, N_e)
            expected = [
                assembly.node_index(N_c, m, n),
                assembly.node_index(N_c, m + 1, n),
                assembly.node_index(N_c, m + 1, n + 1),
                assembly.node_index(N_c, m, n + 1),
            ]
            np.testing.assert_array_equal(idx[:4], expected)


def test_edge_dofs_shared_correctly_between_adjacent_cells():
    """Top edge of cell (m, n) must equal bottom edge of cell (m, n+1);
       right edge of (m, n) must equal left edge of (m+1, n)."""
    N_c, N_e = 4, 2
    N_p = N_e + 1
    for m in range(N_c):
        for n in range(N_c - 1):
            # Top of (m,n) = bottom of (m,n+1)
            a = assembly.cell_global_indices(m, n, N_c, N_e)
            b = assembly.cell_global_indices(m, n + 1, N_c, N_e)
            np.testing.assert_array_equal(
                a[4 + N_p : 4 + 2 * N_p],
                b[4 : 4 + N_p],
            )
    for n in range(N_c):
        for m in range(N_c - 1):
            # Right of (m,n) = left of (m+1,n)
            a = assembly.cell_global_indices(m, n, N_c, N_e)
            b = assembly.cell_global_indices(m + 1, n, N_c, N_e)
            np.testing.assert_array_equal(
                a[4 + 3 * N_p : 4 + 4 * N_p],
                b[4 + 2 * N_p : 4 + 3 * N_p],
            )


def test_all_indices_in_range_and_distinct_per_cell():
    N_c, N_e = 4, 2
    idx = assembly.all_cell_global_indices(N_c, N_e)
    N = assembly.n_total(N_c, N_e)
    assert idx.min() >= 0 and idx.max() < N
    for k in range(idx.shape[0]):
        assert np.unique(idx[k]).size == idx.shape[1]


def test_boundary_dofs_cover_all_domain_boundary():
    """Verify that domain_boundary_dofs() contains exactly the DOFs whose
    owning geometric entity lies on ∂Ω."""
    N_c, N_e = 4, 2
    bdy = set(assembly.domain_boundary_dofs(N_c, N_e).tolist())

    # Corner ∂Ω count: (N_c+1)^2 - (N_c-1)^2 = 4*N_c
    corner_bdy = 4 * N_c
    # Horizontal edges at k=0 or k=N_c: 2·N_c edges × N_p DOFs
    N_p = N_e + 1
    horiz_bdy = 2 * N_c * N_p
    # Vertical edges at m=0 or m=N_c: 2·N_c edges × N_p DOFs
    vert_bdy = 2 * N_c * N_p
    assert len(bdy) == corner_bdy + horiz_bdy + vert_bdy


def test_assemble_global_symmetry_for_symmetric_local():
    """If all K_cells are symmetric, the assembled global A must be symmetric."""
    N_c, N_e = 3, 2
    count = 4 + 4 * (N_e + 1)
    rng = np.random.default_rng(0)
    n_cells = N_c * N_c
    K_cells = rng.standard_normal((n_cells, count, count))
    K_cells = K_cells + K_cells.transpose(0, 2, 1)
    f_cells = rng.standard_normal((n_cells, count))
    cell_idx = assembly.all_cell_global_indices(N_c, N_e)
    N = assembly.n_total(N_c, N_e)
    A, F = assembly.assemble_global(K_cells, f_cells, cell_idx, N)
    A_d = A.toarray()
    np.testing.assert_allclose(A_d, A_d.T, atol=1e-10)
