"""Tests for per-element basis construction."""

import numpy as np

from expmsfem import element_basis
from expmsfem.coefficients import afun_periodic, afun_constant


def test_nodal_basis_partition_of_unity():
    """Sum of the 4 nodal basis functions should be ≡ 1 on the cell (they
    partition unity for the classical MsFEM nodal space)."""
    from expmsfem.local_ops import basefun
    from expmsfem.element_basis import _nodal_basis

    B = basefun(afun_periodic, m=1, n=1, N_c=4, N_f=6)
    nb = _nodal_basis(B, N_f=6)
    total = nb.sum(axis=1)
    np.testing.assert_allclose(total, 1.0, atol=1e-10)


def test_element_basis_shapes():
    N_c, N_f, N_e = 4, 6, 2
    value, B, K_elem, f_elem = element_basis.element_basis(
        afun_periodic, 1, 1, N_c, N_f, N_e
    )
    count = 4 + 4 * (N_e + 1)
    assert value.shape == ((N_f + 1) ** 2, count)
    assert K_elem.shape == (count, count)
    assert f_elem.shape == (count,)


def test_element_stiffness_symmetric_psd():
    N_c, N_f, N_e = 4, 6, 2
    value, B, K_elem, f_elem = element_basis.element_basis(
        afun_periodic, 1, 1, N_c, N_f, N_e
    )
    np.testing.assert_allclose(K_elem, K_elem.T, atol=1e-10)
    eigs = np.linalg.eigvalsh(0.5 * (K_elem + K_elem.T))
    assert eigs.min() > -1e-10


def test_boundary_cell_zero_edges():
    """A corner cell (0,0) has 2 boundary edges → 2·(N_e+1) zero columns."""
    N_c, N_f, N_e = 4, 4, 2
    value, B, K_elem, f_elem = element_basis.element_basis(
        afun_periodic, 0, 0, N_c, N_f, N_e
    )
    # Cell (0,0) has bottom edge on ∂Ω (n=0 → no bottom), and left edge on ∂Ω
    # (m=0 → no left). So columns for bottom and left edges are all zero.
    # Column layout: [4 nodal, bottom block, top block, left block, right block]
    block = N_e + 1
    bottom_cols = value[:, 4 : 4 + block]
    top_cols = value[:, 4 + block : 4 + 2 * block]
    left_cols = value[:, 4 + 2 * block : 4 + 3 * block]
    right_cols = value[:, 4 + 3 * block : 4 + 4 * block]
    np.testing.assert_allclose(bottom_cols, 0, atol=1e-14)
    np.testing.assert_allclose(left_cols, 0, atol=1e-14)
    assert np.linalg.norm(top_cols) > 1e-6
    assert np.linalg.norm(right_cols) > 1e-6


def test_element_basis_spans_nodal_hats_exactly():
    """The 4 nodal basis columns should exactly reproduce the corner values
    {1, 0, 0, 0} etc. at the 4 cell corners — they come from Dirichlet solves
    with hat boundary data, so this is by construction."""
    N_c, N_f, N_e = 4, 4, 1
    value, _, _, _ = element_basis.element_basis(
        afun_periodic, 1, 1, N_c, N_f, N_e
    )
    nb = value[:, :4]
    # Corners of (N_f+1)×(N_f+1) grid
    corners = np.array([0, N_f, (N_f + 1) * (N_f + 1) - 1, N_f * (N_f + 1)])
    # In local_to_global ordering: counterclockwise {(0,0), (Nf,0), (Nf,Nf), (0,Nf)}
    # i.e. [0, N_f, (N_f+1)*N_f+N_f, N_f*(N_f+1)] = [0, N_f, (N_f+1)^2-1, N_f*(N_f+1)]
    for k in range(4):
        # At corner k, the k-th nodal basis should be 1, others 0.
        target = np.zeros(4)
        target[k] = 1
        np.testing.assert_allclose(nb[corners[k], :], target, atol=1e-12)
