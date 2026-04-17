"""Integration test: relative H¹ error decays approximately exponentially as
N_e grows on the periodic coefficient. Uses a small grid so the test runs in
under ~30s on a laptop.
"""

import os

# Serial BLAS inside threaded pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pytest

from expmsfem import local_ops
from expmsfem.coefficients import (
    afun_periodic,
    afun_random,
    afun_highcontrast,
    generate_random_field,
)
from expmsfem.driver import run_expmsfem
from expmsfem.fem import solve_fine


@pytest.mark.slow
def test_convergence_periodic_small():
    N_c, N_f = 8, 16
    u_ref, K_ref, M_ref, _ = solve_fine(afun_periodic, N_c * N_f)
    ws = local_ops.Workspace(afun_periodic, N_c, N_f)
    ws.prefactor_all(n_workers=4)

    h1 = []
    for N_e in range(1, 6):
        out = run_expmsfem(
            afun_periodic, N_c, N_f, N_e,
            u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
            n_workers=4, workspace=ws,
        )
        h1.append(out["e_H1"])

    h1 = np.array(h1)
    # Strictly monotone decreasing
    assert np.all(np.diff(h1) < 0), f"not monotone: {h1}"
    # H1(N_e=1) / H1(N_e=5) should span at least 2 orders of magnitude
    assert h1[0] / h1[-1] > 100, f"decay too slow: {h1}"
    # Slope on log10 should be ≤ -0.5 per mode (roughly exponential)
    log_h1 = np.log10(h1)
    slope = (log_h1[-1] - log_h1[0]) / 4
    assert slope < -0.5, f"slope {slope} not exponentially decaying"


def test_convergence_periodic_tiny():
    """Smaller, always-run version to catch regressions quickly."""
    N_c, N_f = 4, 8
    u_ref, K_ref, M_ref, _ = solve_fine(afun_periodic, N_c * N_f)
    ws = local_ops.Workspace(afun_periodic, N_c, N_f)
    ws.prefactor_all(n_workers=2)

    h1 = []
    for N_e in range(1, 5):
        out = run_expmsfem(
            afun_periodic, N_c, N_f, N_e,
            u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
            n_workers=2, workspace=ws,
        )
        h1.append(out["e_H1"])

    h1 = np.array(h1)
    assert np.all(np.diff(h1) < 0), f"not monotone: {h1}"
    # At least 20x improvement N_e=1 vs N_e=4
    assert h1[0] / h1[-1] > 20, f"decay too slow: {h1}"


def test_convergence_random_tiny():
    """Random heterogeneous field — expect exponential decay similar to
    periodic (both are smoothish multiscale coefficients)."""
    N_c, N_f = 4, 8
    goo = generate_random_field(M=128, seed=0)
    a_fun = afun_random(goo, M=128)

    u_ref, K_ref, M_ref, _ = solve_fine(a_fun, N_c * N_f)
    ws = local_ops.Workspace(a_fun, N_c, N_f)
    ws.prefactor_all(n_workers=2)

    h1 = []
    for N_e in range(1, 5):
        out = run_expmsfem(
            a_fun, N_c, N_f, N_e,
            u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
            n_workers=2, workspace=ws,
        )
        h1.append(out["e_H1"])

    h1 = np.array(h1)
    assert np.all(np.diff(h1) < 0), f"not monotone: {h1}"
    assert h1[0] / h1[-1] > 20, f"decay too slow: {h1}"


def test_convergence_highcontrast_tiny():
    """High-contrast (64×) inclusions — exponential convergence still holds
    but the constant is worse than smooth cases; we just check monotone decay
    and at least 10× improvement over 4 modes."""
    N_c, N_f = 4, 8
    u_ref, K_ref, M_ref, _ = solve_fine(afun_highcontrast, N_c * N_f)
    ws = local_ops.Workspace(afun_highcontrast, N_c, N_f)
    ws.prefactor_all(n_workers=2)

    h1 = []
    for N_e in range(1, 5):
        out = run_expmsfem(
            afun_highcontrast, N_c, N_f, N_e,
            u_ref=u_ref, K_ref=K_ref, M_ref=M_ref,
            n_workers=2, workspace=ws,
        )
        h1.append(out["e_H1"])

    h1 = np.array(h1)
    assert np.all(np.diff(h1) < 0), f"not monotone: {h1}"
    assert h1[0] / h1[-1] > 10, f"decay too slow: {h1}"
