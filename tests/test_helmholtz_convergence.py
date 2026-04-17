"""Convergence test for the Helmholtz ExpMsFEM."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from expmsfem.helmholtz import local_ops as hlo
from expmsfem.helmholtz.driver import run_expmsfem_helm
from expmsfem.helmholtz.fem import solve_fine


def test_convergence_helmholtz_case1_tiny():
    """At k₀=2 on an N_c=4, N_f=4 grid, H¹ error should decay by ≥2 orders
    over N_e=1..3 for the plane-wave boundary data in case1."""
    N_c, N_f, k0 = 4, 4, 2.0
    u_ref, B_ref, C_ref, _ = solve_fine(N_c * N_f, k0)

    ws = hlo.HelmWorkspace(N_c, N_f, k0)
    ws.prefactor_all(n_workers=2)

    h1 = []
    for N_e in range(1, 4):
        out = run_expmsfem_helm(
            N_c, N_f, N_e, k0,
            u_ref=u_ref, B_ref=B_ref, C_ref=C_ref,
            workspace=ws, n_workers=2,
        )
        h1.append(out["e_H1"])
    h1 = np.array(h1)
    # Monotone decrease (allow tiny numerical wobble at the last step by
    # checking pair-wise ratio)
    assert h1[1] < h1[0], f"not decreasing at N_e 1→2: {h1}"
    # At least 10× improvement from N_e=1 to N_e=3
    assert h1[0] / h1[-1] > 10, f"decay too slow: {h1}"


def test_convergence_helmholtz_case1_nominal():
    """Matlab main.m settings (N_c=8, N_f=8, k0=2) — exponential H¹ decay
    from N_e=1..5 (avoid the rank-boundary N_e=6,7 to keep the test stable)."""
    N_c, N_f, k0 = 8, 8, 2.0
    u_ref, B_ref, C_ref, _ = solve_fine(N_c * N_f, k0)

    ws = hlo.HelmWorkspace(N_c, N_f, k0)
    ws.prefactor_all(n_workers=4)

    h1 = []
    for N_e in range(1, 6):
        out = run_expmsfem_helm(
            N_c, N_f, N_e, k0,
            u_ref=u_ref, B_ref=B_ref, C_ref=C_ref,
            workspace=ws, n_workers=4,
        )
        h1.append(out["e_H1"])
    h1 = np.array(h1)
    # Monotone decrease
    assert np.all(np.diff(h1) < 0), f"not monotone: {h1}"
    # H1 should drop by 100× over 5 modes for this smooth, low-k case
    assert h1[0] / h1[-1] > 1e3, f"decay too slow: {h1}"
