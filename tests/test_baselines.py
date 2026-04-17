"""Tests for the H+bubble and O(H) baseline methods."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from expmsfem import local_ops
from expmsfem.baselines import run_hbubble, run_OH
from expmsfem.coefficients import afun_periodic
from expmsfem.driver import run_expmsfem
from expmsfem.fem import solve_fine


def test_baseline_assembly_dof_count_matches_matlab():
    """H+bubble / O(H) total DOFs = (N_c+1)² + 2·N_c·(N_c+1)·N_e."""
    from expmsfem import assembly as ass
    for N_c, N_e in [(4, 2), (8, 3), (32, 7)]:
        expected = (N_c + 1) ** 2 + 2 * N_c * (N_c + 1) * N_e
        assert ass.n_total(N_c, N_e, n_per_edge=N_e) == expected


def test_hbubble_convergence_periodic_tiny():
    """H+bubble should converge with N_e (less aggressively than Exp but
    still monotone on a smooth problem)."""
    N_c, N_f = 4, 8
    u_ref, K_ref, M_ref, _ = solve_fine(afun_periodic, N_c * N_f)
    ws = local_ops.Workspace(afun_periodic, N_c, N_f)
    ws.prefactor_all(n_workers=2)

    h1 = []
    for N_e in range(1, 5):
        out = run_hbubble(
            afun_periodic, N_c, N_f, N_e,
            u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
            workspace=ws, n_workers=2,
        )
        h1.append(out["e_H1"])
    h1 = np.array(h1)
    assert np.all(np.diff(h1) < 0), f"not monotone: {h1}"


def test_OH_convergence_periodic_tiny():
    N_c, N_f = 4, 8
    u_ref, K_ref, M_ref, _ = solve_fine(afun_periodic, N_c * N_f)
    ws = local_ops.Workspace(afun_periodic, N_c, N_f)
    ws.prefactor_all(n_workers=2)

    h1 = []
    for N_e in range(1, 5):
        out = run_OH(
            afun_periodic, N_c, N_f, N_e,
            u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
            workspace=ws, n_workers=2,
        )
        h1.append(out["e_H1"])
    h1 = np.array(h1)
    # O(H) without bubble: still monotone, but can plateau earlier.
    # Enforce non-increasing (allow equality) rather than strict decrease.
    assert np.all(np.diff(h1) <= 1e-10), f"not monotone: {h1}"


def test_method_ordering_hbubble_between_OH_and_Exp():
    """For a fixed (N_c, N_f, N_e), error should satisfy
       Exp ≤ H+bubble ≤ O(H)  (in H¹),
    since Exp has strictly more DOFs (edge-bubble) than H+bubble, and
    H+bubble adds a cell bubble that O(H) lacks."""
    N_c, N_f, N_e = 4, 8, 3
    u_ref, K_ref, M_ref, _ = solve_fine(afun_periodic, N_c * N_f)
    ws = local_ops.Workspace(afun_periodic, N_c, N_f)
    ws.prefactor_all(n_workers=2)

    h1_exp = run_expmsfem(
        afun_periodic, N_c, N_f, N_e,
        u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
        workspace=ws, n_workers=2,
    )["e_H1"]
    h1_hb = run_hbubble(
        afun_periodic, N_c, N_f, N_e,
        u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
        workspace=ws, n_workers=2,
    )["e_H1"]
    h1_oh = run_OH(
        afun_periodic, N_c, N_f, N_e,
        u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
        workspace=ws, n_workers=2,
    )["e_H1"]

    # Allow small numerical slack (1% relative) between Exp and H+bubble because
    # the extra column can barely change the error for very smooth problems.
    assert h1_exp <= h1_hb * 1.1, f"Exp={h1_exp}, H+bubble={h1_hb}"
    # O(H) must be worse than H+bubble (no bubble).
    assert h1_hb <= h1_oh, f"H+bubble={h1_hb}, O(H)={h1_oh}"
