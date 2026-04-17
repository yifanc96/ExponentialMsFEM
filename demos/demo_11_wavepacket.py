"""Time-dependent Schrödinger ExpMsFEM demo: wavepacket scattering through
a **multiscale potential**.

Solves
    i ε ∂_t ψ = −½ ε² Δ ψ + V(x) ψ,  ψ|_∂Ω = 0
on `[0, 1]²` with `ε = 0.2`, backward Euler time step `Δt = 10⁻³`, over
80 steps. The potential

    V(x, y) = V₀ · (sin²(π k_x x / ε) + sin²(π k_y y / ε))

is a periodic "crystal" with oscillations at scale `ε / k_{x,y}` — exactly
the kind of rough, sub-coarse-mesh potential that plain Q1 FEM on the
coarse scale cannot resolve but that ExpMsFEM handles because the edge
eigen-basis adapts to `V` through the shifted cell operator
`−½ε² Δ + V − iε/Δt`.

The figure has five columns. The leftmost panel is `V(x)` itself; the
remaining four are `|ψ(t, x)|²` at four snapshot times. Top row is the
fine-FEM reference, bottom row is the ExpMsFEM reconstruction on a coarse
`N_c = 16, N_f = 8, N_e = 3` mesh. The wavepacket is launched from the
left with momentum along `+x` and diffracts through the lattice.

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
    eps = 0.3
    dt = 1e-3

    # Multiscale periodic "crystal" potential — oscillates at scale ε/kₓ_V = 0.15.
    # V0 is small compared to the wavepacket's kinetic energy kx²/2 = 8 below,
    # so the wavepacket is a *travelling* scattering state, not a bound one.
    V0 = 2.0
    kx_V, ky_V = 2.0, 2.0
    V = lambda x, y: V0 * (np.sin(np.pi * kx_V * x / eps) ** 2
                           + np.sin(np.pi * ky_V * y / eps) ** 2)

    N_c, N_f, N_e = 16, 8, 3
    N_fine = N_c * N_f
    xs = np.linspace(0, 1, N_fine + 1)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    sigma = 0.1
    # Start at the left, moving right with momentum kx = 4 → group velocity 4.
    x0, y0, kx = 0.2, 0.5, 4.0
    psi0 = (np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
            * np.exp(1j * kx * X / eps)).ravel()

    param = SemiclassicalParam(eps=eps, V_fun=V, dt=dt)
    # 150 steps × Δt = 0.15; packet crosses ~0.6 of the domain.
    n_steps = 150
    save_stride = 30

    print(f"[demo_11] wavepacket scattering through a multiscale V "
          f"(ε={eps}, Δt={dt}, N_c={N_c}, N_f={N_f}, N_e={N_e})")
    print("[demo_11] fine reference ...")
    ts_ref, frames_ref, B, M = solve_fine_backward_euler(
        param, psi0, N_fine, n_steps, save_stride=save_stride,
    )
    print("[demo_11] ExpMsFEM ...")
    ts_ms, frames_ms, _ = run_expmsfem_schrodinger(
        param, psi0, N_c, N_f, N_e, n_steps, save_stride=save_stride, n_workers=4,
    )

    for i, t in enumerate(ts_ms):
        e = frames_ref[:, i] - frames_ms[:, i]
        rel = float(np.sqrt(
            np.real(e.conj() @ (M @ e))
            / np.real(frames_ref[:, i].conj() @ (M @ frames_ref[:, i]))
        ))
        print(f"  t = {t:.3f}  rel L2 = {rel:.2e}")

    # Six columns: V(x) + 5 snapshot times at even spacing
    snap_indices = [0, 1, 2, 3, 4, 5]
    V_grid = V(X, Y)
    fig, axes = plt.subplots(2, 6, figsize=(22, 7.5))

    # Column 0: V(x) (same in both rows — just show once)
    for row in range(2):
        ax = axes[row, 0]
        im = ax.pcolormesh(X, Y, V_grid, cmap="viridis", shading="auto")
        ax.set_title(f"V(x)  (V₀={V0}, period ε/2)")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    # Columns 1..5: |ψ|² snapshots, top = fine FEM, bottom = ExpMsFEM
    for col_offset, idx in enumerate(snap_indices[1:]):
        col = col_offset + 1
        t = ts_ms[idx]
        ref_sq = np.abs(frames_ref[:, idx].reshape(N_fine + 1, N_fine + 1)) ** 2
        ms_sq = np.abs(frames_ms[:, idx].reshape(N_fine + 1, N_fine + 1)) ** 2
        vmax = max(ref_sq.max(), ms_sq.max())

        for row, Z, label in [(0, ref_sq, "fine FEM"),
                              (1, ms_sq, "ExpMsFEM")]:
            ax = axes[row, col]
            im = ax.pcolormesh(X, Y, Z, cmap="magma", shading="auto",
                               vmin=0, vmax=vmax)
            ax.set_title(f"{label}  |ψ|², t = {t:.3f}")
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
            ax.set_aspect("equal")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

    fig.suptitle(
        "Time-dependent Schrödinger with a multiscale potential: "
        f"wavepacket scattering (ε={eps}, Δt={dt})",
        y=1.01, fontsize=13,
    )
    save(fig, "wavepacket.png")


if __name__ == "__main__":
    main()
