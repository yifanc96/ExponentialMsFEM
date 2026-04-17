"""Time-dependent Schrödinger ExpMsFEM demo: Gaussian wavepacket propagation.

Solves
    i ε ∂_t ψ = -½ ε² Δ ψ + V(x) ψ,  ψ|_∂Ω = 0
on `[0, 1]²` with `ε = 0.2`, free particle (V = 0), and a localised
Gaussian wavepacket initial condition

    ψ₀(x, y) = exp(-((x - 0.3)² + (y - 0.5)²) / (2 σ²)) exp(i k x / ε).

Time integration: backward Euler with `Δt = 10⁻³`, 50 steps. The figure
shows 4 snapshots (`|ψ|²`) of the ExpMsFEM reconstruction side by side
with the fine-FEM reference. The packet translates to the right and
disperses; backward Euler dissipates high frequencies modestly.

Writes figures/wavepacket.png.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from _util import set_style, save
from expmsfem.schrodinger.time_dep import (
    SemiclassicalParam, solve_fine_backward_euler, run_expmsfem_schrodinger,
)


def main():
    set_style()
    eps = 0.2
    dt = 1e-3
    V = lambda x, y: np.zeros_like(np.asarray(x) * np.asarray(y))

    N_c, N_f, N_e = 16, 8, 3
    N_fine = N_c * N_f
    xs = np.linspace(0, 1, N_fine + 1)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    sigma = 0.1
    x0, y0, kx = 0.3, 0.5, 3.0
    psi0 = (np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
            * np.exp(1j * kx * X / eps)).ravel()

    param = SemiclassicalParam(eps=eps, V_fun=V, dt=dt)
    n_steps = 50

    print("[demo_11] fine reference ...")
    ts_ref, frames_ref, B, M = solve_fine_backward_euler(
        param, psi0, N_fine, n_steps, save_stride=10,
    )
    print(f"[demo_11] ExpMsFEM  (N_c={N_c}, N_f={N_f}, N_e={N_e}) ...")
    ts_ms, frames_ms, _ = run_expmsfem_schrodinger(
        param, psi0, N_c, N_f, N_e, n_steps, save_stride=10, n_workers=4,
    )

    # Print L² error for each snapshot
    for i, t in enumerate(ts_ms):
        e = frames_ref[:, i] - frames_ms[:, i]
        rel = float(np.sqrt(
            np.real(e.conj() @ (M @ e))
            / np.real(frames_ref[:, i].conj() @ (M @ frames_ref[:, i]))
        ))
        print(f"  t = {t:.3f}  rel L2 = {rel:.2e}")

    # Render |ψ|² snapshots for 4 selected times
    snap_indices = [0, 2, 4, 5]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for col, idx in enumerate(snap_indices):
        t = ts_ms[idx]
        ref_sq = np.abs(frames_ref[:, idx].reshape(N_fine + 1, N_fine + 1)) ** 2
        ms_sq = np.abs(frames_ms[:, idx].reshape(N_fine + 1, N_fine + 1)) ** 2
        vmax = max(ref_sq.max(), ms_sq.max())

        ax_ref = axes[0, col]
        im = ax_ref.pcolormesh(X, Y, ref_sq, cmap="magma",
                               shading="auto", vmin=0, vmax=vmax)
        ax_ref.set_title(f"fine FEM  |ψ|², t = {t:.3f}")
        plt.colorbar(im, ax=ax_ref, fraction=0.045, pad=0.03)
        ax_ref.set_aspect("equal")
        ax_ref.set_xlabel("$x_1$")
        ax_ref.set_ylabel("$x_2$")

        ax_ms = axes[1, col]
        im = ax_ms.pcolormesh(X, Y, ms_sq, cmap="magma",
                              shading="auto", vmin=0, vmax=vmax)
        ax_ms.set_title(f"ExpMsFEM  |ψ|², t = {t:.3f}")
        plt.colorbar(im, ax=ax_ms, fraction=0.045, pad=0.03)
        ax_ms.set_aspect("equal")
        ax_ms.set_xlabel("$x_1$")
        ax_ms.set_ylabel("$x_2$")

    fig.suptitle(
        f"Time-dependent Schrödinger: Gaussian wavepacket "
        f"(ε={eps}, Δt={dt}, N_c={N_c}, N_f={N_f}, N_e={N_e})",
        y=1.01, fontsize=13,
    )
    save(fig, "wavepacket.png")


if __name__ == "__main__":
    main()
