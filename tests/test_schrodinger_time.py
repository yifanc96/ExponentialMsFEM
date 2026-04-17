"""Tests for the time-dependent Schrödinger ExpMsFEM propagator."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pytest

from expmsfem.schrodinger.time_dep import (
    SemiclassicalParam, SchrodingerPropagator, solve_fine_backward_euler,
    run_expmsfem_schrodinger,
)


def _make_ic(N_fine: int, x0=0.5, y0=0.5, sigma=0.1, kx=0.0, ky=0.0,
             eps=0.5):
    xs = np.linspace(0, 1, N_fine + 1)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    psi = (np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
           * np.exp(1j * (kx * X + ky * Y) / eps))
    return psi.astype(np.complex128).ravel()


def test_fine_backward_euler_runs():
    """Smoke test: fine backward-Euler propagates without exploding and
    decays the L² norm slightly (expected BE dissipation)."""
    eps = 0.5
    V = lambda x, y: np.zeros_like(np.asarray(x) * np.asarray(y))
    param = SemiclassicalParam(eps=eps, V_fun=V, dt=1e-2)
    psi0 = _make_ic(32, eps=eps)
    ts, frames, B, M = solve_fine_backward_euler(param, psi0, N_f=32,
                                                  n_steps=5, save_stride=1)
    norms = np.array([
        float(np.real(frames[:, i].conj() @ (M @ frames[:, i])))
        for i in range(frames.shape[1])
    ])
    # Monotone-decreasing norms (backward Euler dissipates) but not to zero
    assert np.all(np.diff(norms) <= 0)
    assert norms[-1] > 0.5 * norms[0]


def test_expmsfem_converges_in_N_e():
    """On a simple real Gaussian free-evolution problem, the ExpMsFEM L²
    error drops monotonically with `N_e`."""
    eps = 0.5
    V = lambda x, y: np.zeros_like(np.asarray(x) * np.asarray(y))
    param = SemiclassicalParam(eps=eps, V_fun=V, dt=1e-2)
    N_c, N_f = 8, 8
    N_fine = N_c * N_f
    psi0 = _make_ic(N_fine, kx=0.0, eps=eps)

    # Fine reference
    ts_ref, frames_ref, B_ref, M_ref = solve_fine_backward_euler(
        param, psi0, N_fine, n_steps=5, save_stride=5,
    )

    errs = []
    for N_e in [1, 3, 5]:
        ts_ms, frames_ms, prop = run_expmsfem_schrodinger(
            param, psi0, N_c, N_f, N_e=N_e, n_steps=5, save_stride=5,
            n_workers=2,
        )
        psi_r = frames_ref[:, -1]
        psi_m = frames_ms[:, -1]
        e = psi_r - psi_m
        rel = float(np.sqrt(
            np.real(e.conj() @ (M_ref @ e))
            / np.real(psi_r.conj() @ (M_ref @ psi_r))
        ))
        errs.append(rel)
    errs = np.array(errs)
    assert np.all(np.diff(errs) < 0), f"not monotone: {errs}"
    # From N_e=1 to N_e=5, expect ≥ 2 orders of magnitude improvement
    assert errs[0] / errs[-1] > 1e2, f"errs = {errs}"


def test_wavepacket_stays_bounded():
    """A non-trivial wavepacket stays close to the fine reference across
    many steps (no blow-up)."""
    eps = 0.2
    V = lambda x, y: np.zeros_like(np.asarray(x) * np.asarray(y))
    param = SemiclassicalParam(eps=eps, V_fun=V, dt=1e-3)
    N_c, N_f = 8, 8
    N_fine = N_c * N_f
    psi0 = _make_ic(N_fine, x0=0.3, y0=0.5, kx=3.0, eps=eps)

    ts_ref, frames_ref, B_ref, M_ref = solve_fine_backward_euler(
        param, psi0, N_fine, n_steps=30, save_stride=10,
    )
    ts_ms, frames_ms, prop = run_expmsfem_schrodinger(
        param, psi0, N_c, N_f, N_e=3, n_steps=30, save_stride=10,
        n_workers=2,
    )

    # All saved snapshots within 5% of the fine reference
    for i in range(frames_ref.shape[1]):
        e = frames_ref[:, i] - frames_ms[:, i]
        rel = float(np.sqrt(
            np.real(e.conj() @ (M_ref @ e))
            / np.real(frames_ref[:, i].conj() @ (M_ref @ frames_ref[:, i]))
        ))
        assert rel < 5e-2, f"frame {i}: rel = {rel}"
